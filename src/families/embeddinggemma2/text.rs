//! EmbeddingGemma 2 text encoder (`EmbeddingGemma2TextModel`, transformers 5.22.2).
//!
//! candle's `gemma4::text` is the causal Gemma 4 decoder, so this is written from the transformers
//! source. What differs from a Gemma 4 decoder layer:
//! - attention is bidirectional; sliding layers see `|q - kv| <= sliding_window`;
//! - attention scale is 1.0, with q/k RMSNorm (scaled) and v RMSNorm (unscaled);
//! - full-attention layers use their own head_dim and kv-head count (`per_layer_config`);
//! - per-layer embeddings are projection-only: `RMSNorm(W inputs_embeds / sqrt(hidden))`, one 512-d slice
//!   per layer, mixed in by a gated bottleneck after the MLP; each layer ends with `* layer_scalar`;
//! - the final norm is followed by `embedding_projection` (512 -> 768) on every token.

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Linear, VarBuilder};

use super::config::{LayerShape, TextConfig};
use crate::attention::{attention, heads_first, padding_mask};
use crate::mlp::{Activation, GatedMlp};
use crate::norm::RmsNorm;
use crate::rope;
use crate::weights::linear_no_bias as linear;

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    v_norm: RmsNorm,
    num_heads: usize,
    shape: LayerShape,
}

impl Attention {
    fn new(cfg: &TextConfig, shape: LayerShape, vb: VarBuilder) -> candle_core::Result<Self> {
        let (h, hd, kv, hs) = (
            cfg.num_attention_heads,
            shape.head_dim,
            shape.num_kv_heads,
            cfg.hidden_size,
        );
        Ok(Self {
            q_proj: linear(hs, h * hd, vb.pp("q_proj"))?,
            k_proj: linear(hs, kv * hd, vb.pp("k_proj"))?,
            v_proj: linear(hs, kv * hd, vb.pp("v_proj"))?,
            o_proj: linear(h * hd, hs, vb.pp("o_proj"))?,
            q_norm: RmsNorm::new(hd, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: RmsNorm::new(hd, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            v_norm: RmsNorm::without_scale(hd, cfg.rms_norm_eps, &vb)?,
            num_heads: h,
            shape,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let (b, s, _) = xs.dims3()?;
        let hd = self.shape.head_dim;
        // Norms on the contiguous (b, s, heads, hd) layout, then one transpose copy per tensor.
        let heads = |t: Tensor, n: usize| t.reshape((b, s, n, hd));
        let kv = self.shape.num_kv_heads;
        let q = self
            .q_norm
            .forward(&heads(self.q_proj.forward(xs)?, self.num_heads)?)?;
        let k = self.k_norm.forward(&heads(self.k_proj.forward(xs)?, kv)?)?;
        let v = self.v_norm.forward(&heads(self.v_proj.forward(xs)?, kv)?)?;
        let q = rope::apply(&heads_first(&q)?, cos, sin)?;
        let k = rope::apply(&heads_first(&k)?, cos, sin)?;
        let v = heads_first(&v)?;
        let out = attention(&q, &k, &v, Some(mask))?;
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
    ple_gate: Linear,
    ple_proj: Linear,
    ple_norm: RmsNorm,
    layer_scalar: Tensor,
    act: Activation,
}

impl Layer {
    fn new(
        cfg: &TextConfig,
        shape: LayerShape,
        act: Activation,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let (h, i, p, eps) = (
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.hidden_size_per_layer_input,
            cfg.rms_norm_eps,
        );
        let mlp = vb.pp("mlp");
        let ple = vb.pp("ple_block");
        Ok(Self {
            attn: Attention::new(cfg, shape, vb.pp("self_attn"))?,
            mlp: GatedMlp {
                gate: linear(h, i, mlp.pp("gate_proj"))?,
                up: linear(h, i, mlp.pp("up_proj"))?,
                down: linear(i, h, mlp.pp("down_proj"))?,
                act,
            },
            input_ln: RmsNorm::new(h, eps, vb.pp("input_layernorm"))?,
            post_attn_ln: RmsNorm::new(h, eps, vb.pp("post_attention_layernorm"))?,
            pre_ff_ln: RmsNorm::new(h, eps, vb.pp("pre_feedforward_layernorm"))?,
            post_ff_ln: RmsNorm::new(h, eps, vb.pp("post_feedforward_layernorm"))?,
            ple_gate: linear(h, p, ple.pp("per_layer_input_gate"))?,
            ple_proj: linear(p, h, ple.pp("per_layer_projection"))?,
            ple_norm: RmsNorm::new(h, eps, ple.pp("post_per_layer_input_norm"))?,
            layer_scalar: vb.get(1, "layer_scalar")?,
            act,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        per_layer_input: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let h = self
            .attn
            .forward(&self.input_ln.forward(xs)?, cos, sin, mask)?;
        let xs = (xs + self.post_attn_ln.forward(&h)?)?;
        let n = self.pre_ff_ln.forward(&xs)?;
        let h = self.mlp.forward(&n)?;
        let xs = (xs + self.post_ff_ln.forward(&h)?)?;
        // EmbeddingGemma2TextPLEBlock
        let g = (self.act.apply(&self.ple_gate.forward(&xs)?)? * per_layer_input)?;
        let xs = (&xs + self.ple_norm.forward(&self.ple_proj.forward(&g)?)?)?;
        xs.broadcast_mul(&self.layer_scalar)
    }
}

pub struct TextEncoder {
    layers: Vec<Layer>,
    shapes: Vec<LayerShape>,
    ple_proj: Linear,
    ple_norm: RmsNorm,
    norm: RmsNorm,
    out_proj: Linear,
    hidden: usize,
    per_layer: usize,
    sliding_window: usize,
    dtype: DType,
    device: Device,
}

impl TextEncoder {
    /// `vb` points at `language_model`.
    pub fn new(cfg: &TextConfig, act: Activation, vb: VarBuilder) -> Result<Self> {
        let shapes = (0..cfg.num_hidden_layers)
            .map(|i| cfg.layer_shape(i))
            .collect::<Result<Vec<_>>>()?;
        let lv = vb.pp("layers");
        let layers = shapes
            .iter()
            .enumerate()
            .map(|(i, s)| {
                Layer::new(cfg, *s, act, lv.pp(i)).with_context(|| format!("text layer {i}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let n = cfg.num_hidden_layers;
        let p = cfg.hidden_size_per_layer_input;
        Ok(Self {
            layers,
            shapes,
            ple_proj: linear(
                cfg.hidden_size,
                n * p,
                vb.pp("ple").pp("per_layer_model_projection"),
            )?,
            ple_norm: RmsNorm::new(
                p,
                cfg.rms_norm_eps,
                vb.pp("ple").pp("per_layer_projection_norm"),
            )?,
            norm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?,
            out_proj: linear(
                cfg.hidden_size,
                cfg.embedding_dim,
                vb.pp("embedding_projection"),
            )?,
            hidden: cfg.hidden_size,
            per_layer: p,
            sliding_window: cfg.sliding_window,
            dtype: vb.dtype(),
            device: vb.device().clone(),
        })
    }

    /// `embeds`: (b, s, hidden) right-padded input embeddings (token rows already scaled by
    /// sqrt(hidden), soft tokens merged in). Returns the projected last hidden states (b, s, 768) f32;
    /// the core pools them (mean over `lengths`) and normalizes.
    pub fn forward(&self, embeds: &Tensor, lengths: &[usize]) -> Result<Tensor> {
        let (b, s, _) = embeds.dims3()?;
        if lengths.len() != b || lengths.iter().any(|&l| l == 0 || l > s) {
            bail!("bad lengths {lengths:?} for batch ({b}, {s})");
        }
        let xs = embeds.to_dtype(self.dtype)?;
        let n = self.layers.len();
        // EmbeddingGemma2TextPLE
        let ple = (self.ple_proj.forward(&xs)? * (self.hidden as f64).powf(-0.5))?;
        let ple = self
            .ple_norm
            .forward(&ple.reshape((b, s, n, self.per_layer))?)?;

        let full_mask = padding_mask(lengths, s, None, &self.device)?;
        let sliding_mask = if s > self.sliding_window + 1 {
            padding_mask(lengths, s, Some(self.sliding_window), &self.device)?
        } else {
            full_mask.clone()
        };
        let mut ropes: Vec<((usize, u64), (Tensor, Tensor))> = Vec::new();
        let mut xs = xs;
        for (i, (layer, shape)) in self.layers.iter().zip(&self.shapes).enumerate() {
            let key = (shape.head_dim, shape.rope_theta.to_bits());
            if !ropes.iter().any(|(k, _)| *k == key) {
                let tables = rope::tables_1d(
                    s,
                    shape.head_dim,
                    shape.rope_theta,
                    self.dtype,
                    &self.device,
                )?;
                ropes.push((key, tables));
            }
            let (cos, sin) = &ropes.iter().find(|(k, _)| *k == key).unwrap().1;
            let mask = if shape.sliding {
                &sliding_mask
            } else {
                &full_mask
            };
            let pli = ple.narrow(2, i, 1)?.squeeze(2)?;
            xs = layer.forward(&xs, &pli, cos, sin, mask)?;
        }
        Ok(self
            .out_proj
            .forward(&self.norm.forward(&xs)?)?
            .to_dtype(DType::F32)?)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}
