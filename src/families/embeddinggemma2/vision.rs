//! Gemma 4 vision tower (`Gemma4VisionModel`) and the `embed_vision` projection.
//!
//! Adapted from candle-transformers 0.11.1 `models/gemma4/vision.rs` (Apache-2.0 / MIT) and checked
//! against transformers 5.22.2 `modeling_gemma4.py`. Differences from the candle version, which
//! cannot load this checkpoint as is:
//! - RMSNorm multiplies by `weight`, not `1 + weight` (the stored q_norm weights are ~1.0).
//! - Linears are `Gemma4ClippableLinear`, so weights live at `<name>.linear.weight`.
//! - Pooling and the `sqrt(hidden)` scale run in f32, as transformers does to stay in range.
//! - Images with the same patch grid run as one batch.

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};

use super::config::VisionConfig;
use crate::attention::{attention_fused, heads_first};
use crate::mlp::{Activation, GatedMlp};
use crate::norm::RmsNorm;
use crate::weights::linear_no_bias;
use crate::{pool, rope};

/// Weight map: a `Gemma4ClippableLinear` keeps its weight at `<name>.linear.weight`.
fn clippable(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    linear_no_bias(in_dim, out_dim, vb.pp("linear"))
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    v_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let (h, kv, hd, hs) = (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            cfg.hidden_size,
        );
        Ok(Self {
            q_proj: clippable(hs, h * hd, vb.pp("q_proj"))?,
            k_proj: clippable(hs, kv * hd, vb.pp("k_proj"))?,
            v_proj: clippable(hs, kv * hd, vb.pp("v_proj"))?,
            o_proj: clippable(h * hd, hs, vb.pp("o_proj"))?,
            q_norm: RmsNorm::new(hd, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: RmsNorm::new(hd, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            v_norm: RmsNorm::without_scale(hd, cfg.rms_norm_eps, &vb)?,
            num_heads: h,
            num_kv_heads: kv,
            head_dim: hd,
        })
    }

    /// `cos`/`sin`: (2, s, head_dim / 4), one table per spatial axis.
    fn forward(&self, xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (b, s, _) = xs.dims3()?;
        let hd = self.head_dim;
        // Norms run on the contiguous (b, s, heads, hd) layout, before the head transpose.
        let heads = |t: Tensor, n: usize| t.reshape((b, s, n, hd));
        let q = self
            .q_norm
            .forward(&heads(self.q_proj.forward(xs)?, self.num_heads)?)?;
        let k = self
            .k_norm
            .forward(&heads(self.k_proj.forward(xs)?, self.num_kv_heads)?)?;
        let v = self
            .v_norm
            .forward(&heads(self.v_proj.forward(xs)?, self.num_kv_heads)?)?;
        let q = rope::apply_axial_2d(&q, cos, sin)?;
        let k = rope::apply_axial_2d(&k, cos, sin)?;
        let v = heads_first(&v)?;
        let out = attention_fused(&q, &k, &v)?;
        out.transpose(1, 2)?
            .reshape((b, s, self.num_heads * hd))?
            .apply(&self.o_proj)
    }
}

struct Layer {
    attn: Attention,
    mlp: GatedMlp,
    input_ln: RmsNorm,
    post_attn_ln: RmsNorm,
    pre_ff_ln: RmsNorm,
    post_ff_ln: RmsNorm,
}

impl Layer {
    fn new(cfg: &VisionConfig, act: Activation, vb: VarBuilder) -> Result<Self> {
        let (h, i, eps) = (cfg.hidden_size, cfg.intermediate_size, cfg.rms_norm_eps);
        let mlp = vb.pp("mlp");
        Ok(Self {
            attn: Attention::new(cfg, vb.pp("self_attn"))?,
            mlp: GatedMlp {
                gate: clippable(h, i, mlp.pp("gate_proj"))?,
                up: clippable(h, i, mlp.pp("up_proj"))?,
                down: clippable(i, h, mlp.pp("down_proj"))?,
                act,
            },
            input_ln: RmsNorm::new(h, eps, vb.pp("input_layernorm"))?,
            post_attn_ln: RmsNorm::new(h, eps, vb.pp("post_attention_layernorm"))?,
            pre_ff_ln: RmsNorm::new(h, eps, vb.pp("pre_feedforward_layernorm"))?,
            post_ff_ln: RmsNorm::new(h, eps, vb.pp("post_feedforward_layernorm"))?,
        })
    }

    fn forward(&self, xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let h = self.attn.forward(&self.input_ln.forward(xs)?, cos, sin)?;
        let xs = (xs + self.post_attn_ln.forward(&h)?)?;
        let h = self.mlp.forward(&self.pre_ff_ln.forward(&xs)?)?;
        xs + self.post_ff_ln.forward(&h)?
    }
}

/// Vision tower plus `embed_vision`: patches in, text-width soft tokens out.
pub struct VisionTower {
    input_proj: Linear,
    pos_table_x: Tensor,
    pos_table_y: Tensor,
    layers: Vec<Layer>,
    embed_norm: RmsNorm,
    embed_proj: Linear,
    inv_freq: Vec<f32>,
    hidden: usize,
    pool: usize,
    dtype: DType,
    device: Device,
}

impl VisionTower {
    /// `vb_tower` points at `vision_tower`, `vb_embed` at `embed_vision`.
    pub fn new(
        cfg: &VisionConfig,
        act: Activation,
        text_hidden: usize,
        vb_tower: VarBuilder,
        vb_embed: VarBuilder,
    ) -> Result<Self> {
        let pe = vb_tower.pp("patch_embedder");
        let ps = cfg.patch_size;
        let input_proj = linear_no_bias(3 * ps * ps, cfg.hidden_size, pe.pp("input_proj"))?;
        let table = pe.get(
            (2, cfg.position_embedding_size, cfg.hidden_size),
            "position_embedding_table",
        )?;
        let enc = vb_tower.pp("encoder").pp("layers");
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| Layer::new(cfg, act, enc.pp(i)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            input_proj,
            pos_table_x: table.i(0)?.contiguous()?,
            pos_table_y: table.i(1)?.contiguous()?,
            layers,
            embed_norm: RmsNorm::without_scale(cfg.hidden_size, cfg.rms_norm_eps, &vb_embed)?,
            embed_proj: linear_no_bias(
                cfg.hidden_size,
                text_hidden,
                vb_embed.pp("embedding_projection"),
            )?,
            inv_freq: rope::axial_inv_freq(cfg.head_dim, cfg.rope_parameters.rope_theta),
            hidden: cfg.hidden_size,
            pool: cfg.pooling_kernel_size,
            dtype: vb_tower.dtype(),
            device: vb_tower.device().clone(),
        })
    }

    /// Patch positions in processor order (row-major), as (x, y).
    fn positions(grid_w: usize, grid_h: usize) -> (Vec<u32>, Vec<u32>) {
        let n = grid_w * grid_h;
        let xs = (0..n).map(|i| (i % grid_w) as u32).collect();
        let ys = (0..n).map(|i| (i / grid_w) as u32).collect();
        (xs, ys)
    }

    /// Encode `b` images that share one patch grid.
    ///
    /// `pixels`: (b, grid_h * grid_w, 3 * patch^2) f32 in [0, 1]. Returns (b, soft_tokens, text_hidden)
    /// in the model dtype.
    pub fn forward(&self, pixels: &Tensor, grid_w: usize, grid_h: usize) -> Result<Tensor> {
        let (xs, ys) = Self::positions(grid_w, grid_h);
        // Gemma4VisionPatchEmbedder: 2 * (x - 0.5), cast, project, add the x and y position rows.
        let px = ((pixels.to_device(&self.device)?.to_dtype(DType::F32)? - 0.5)? * 2.0)?
            .to_dtype(self.dtype)?;
        let h = self.input_proj.forward(&px)?;
        let xi = Tensor::new(xs.as_slice(), &self.device)?;
        let yi = Tensor::new(ys.as_slice(), &self.device)?;
        let pos =
            (self.pos_table_x.index_select(&xi, 0)? + self.pos_table_y.index_select(&yi, 0)?)?;
        let mut h = h.broadcast_add(&pos.unsqueeze(0)?)?;

        let (cos, sin) = rope::tables_axial_2d(&xs, &ys, &self.inv_freq, self.dtype, &self.device)?;
        for layer in &self.layers {
            h = layer.forward(&h, &cos, &sin)?;
        }

        // Gemma4VisionPooler: k x k average in f32 by (x // k, y // k), then * sqrt(hidden), as one
        // f32 product.
        let k = self.pool;
        let pooled = pool::grid_sum(&h.to_dtype(DType::F32)?, grid_w, grid_h, k)?;
        let pooled = (pooled * ((self.hidden as f64).sqrt() / (k * k) as f64))?;
        // Back to the working dtype (Gemma4VisionModel.forward), then embed_vision.
        let soft = self.embed_norm.forward(&pooled.to_dtype(self.dtype)?)?;
        self.embed_proj.forward(&soft)
    }
}
