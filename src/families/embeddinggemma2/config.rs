//! The parts of `config.json` (`EmbeddingGemma2Config`) that the text and vision towers need.
//! Activations are checked by `crate::mlp::Activation::from_config` at load.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
    pub image_token_id: u32,
    pub boi_token_id: u32,
    pub eoi_token_id: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeTheta {
    pub rope_theta: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextRope {
    pub full_attention: RopeTheta,
    pub sliding_attention: RopeTheta,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LayerOverride {
    pub head_dim: Option<usize>,
    pub num_key_value_heads: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub hidden_activation: String,
    pub hidden_size_per_layer_input: usize,
    pub embedding_dim: usize,
    pub layer_types: Vec<String>,
    /// Keys are zero-padded layer indices ("05"). Absent: full layers use 512 / 1 kv head,
    /// as `EmbeddingGemma2TextConfig.__post_init__` does.
    #[serde(default)]
    pub per_layer_config: Option<HashMap<String, LayerOverride>>,
    pub rope_parameters: TextRope,
    pub rms_norm_eps: f64,
    pub sliding_window: usize,
    pub vocab_size: usize,
    pub pad_token_id: u32,
}

/// Resolved attention shape of one text layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerShape {
    pub sliding: bool,
    pub head_dim: usize,
    pub num_kv_heads: usize,
    pub rope_theta: f64,
}

impl TextConfig {
    pub fn layer_shape(&self, idx: usize) -> Result<LayerShape> {
        let kind = self
            .layer_types
            .get(idx)
            .with_context(|| format!("no layer_types entry for layer {idx}"))?;
        let sliding = match kind.as_str() {
            "sliding_attention" => true,
            "full_attention" => false,
            other => bail!("unknown layer type {other:?} for layer {idx}"),
        };
        let (mut head_dim, mut num_kv_heads) = (self.head_dim, self.num_key_value_heads);
        match &self.per_layer_config {
            Some(map) => {
                let o = map
                    .iter()
                    .find(|(k, _)| k.parse::<usize>().ok() == Some(idx))
                    .map(|(_, v)| v);
                if let Some(o) = o {
                    head_dim = o.head_dim.unwrap_or(head_dim);
                    num_kv_heads = o.num_key_value_heads.unwrap_or(num_kv_heads);
                }
            }
            None if !sliding => {
                head_dim = 512;
                num_kv_heads = 1;
            }
            None => {}
        }
        let rope_theta = if sliding {
            self.rope_parameters.sliding_attention.rope_theta
        } else {
            self.rope_parameters.full_attention.rope_theta
        };
        Ok(LayerShape {
            sliding,
            head_dim,
            num_kv_heads,
            rope_theta,
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct VisionRope {
    pub rope_theta: f64,
    pub rope_type: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub hidden_activation: String,
    pub patch_size: usize,
    pub pooling_kernel_size: usize,
    pub position_embedding_size: usize,
    pub default_output_length: usize,
    pub rms_norm_eps: f64,
    pub rope_parameters: VisionRope,
    pub standardize: bool,
    #[serde(default)]
    pub use_clipped_linears: bool,
}

impl Config {
    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        let raw =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let cfg: Config =
            serde_json::from_str(&raw).with_context(|| format!("parse {}", path.display()))?;
        cfg.check()?;
        Ok(cfg)
    }

    /// Refuse configs this port does not implement, rather than computing something else.
    fn check(&self) -> Result<()> {
        let v = &self.vision_config;
        if v.standardize {
            bail!("vision_config.standardize = true is not implemented");
        }
        if v.use_clipped_linears {
            bail!("vision_config.use_clipped_linears = true is not implemented");
        }
        if v.rope_parameters.rope_type != "axial" {
            bail!(
                "vision rope_type {:?} is not implemented",
                v.rope_parameters.rope_type
            );
        }
        let t = &self.text_config;
        if t.layer_types.len() != t.num_hidden_layers {
            bail!(
                "layer_types has {} entries for {} layers",
                t.layer_types.len(),
                t.num_hidden_layers
            );
        }
        for i in 0..t.num_hidden_layers {
            let s = t.layer_shape(i)?;
            if !t.num_attention_heads.is_multiple_of(s.num_kv_heads) {
                bail!(
                    "layer {i}: {} heads not a multiple of {} kv heads",
                    t.num_attention_heads,
                    s.num_kv_heads
                );
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_config(per_layer: &str) -> TextConfig {
        let json = format!(
            r#"{{"hidden_size": 512, "intermediate_size": 2048, "num_hidden_layers": 4,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 256,
            "hidden_activation": "gelu_pytorch_tanh", "hidden_size_per_layer_input": 512,
            "embedding_dim": 768, "rms_norm_eps": 1e-6, "sliding_window": 512,
            "vocab_size": 262144, "pad_token_id": 0,
            "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
            "rope_parameters": {{"full_attention": {{"rope_theta": 1000000.0}},
                                 "sliding_attention": {{"rope_theta": 10000.0}}}}
            {per_layer}}}"#
        );
        serde_json::from_str(&json).unwrap()
    }

    #[test]
    fn per_layer_config_overrides_full_layers() {
        let cfg = text_config(
            r#", "per_layer_config": {"01": {"head_dim": 512, "num_key_value_heads": 1}}"#,
        );
        let s0 = cfg.layer_shape(0).unwrap();
        let s1 = cfg.layer_shape(1).unwrap();
        let s3 = cfg.layer_shape(3).unwrap();
        assert_eq!(
            (s0.sliding, s0.head_dim, s0.num_kv_heads, s0.rope_theta),
            (true, 256, 2, 1e4)
        );
        assert_eq!(
            (s1.sliding, s1.head_dim, s1.num_kv_heads, s1.rope_theta),
            (false, 512, 1, 1e6)
        );
        // Not listed: keeps the defaults.
        assert_eq!((s3.head_dim, s3.num_kv_heads), (256, 2));
    }

    #[test]
    fn missing_per_layer_config_uses_global_defaults() {
        let cfg = text_config("");
        let s1 = cfg.layer_shape(1).unwrap();
        assert_eq!((s1.head_dim, s1.num_kv_heads), (512, 1));
        assert_eq!(cfg.layer_shape(0).unwrap().head_dim, 256);
        assert!(cfg.layer_shape(4).is_err());
    }
}
