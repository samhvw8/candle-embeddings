//! RMSNorm.

use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;

/// `x * (mean(x^2) + eps)^-0.5 * weight`, with the weight used as stored.
///
/// Runs candle's fused kernel, which accumulates in f32 for every dtype, as transformers does.
/// Gemma 3 and candle's gemma4 vision tower multiply by `1 + weight` instead; a family with that
/// convention folds the 1 into the weight at load or adds a variant here.
#[derive(Debug, Clone)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f32,
}

impl RmsNorm {
    /// Weight at `<vb>.weight`, shape `(dim,)`.
    pub fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get(dim, "weight")?,
            eps: eps as f32,
        })
    }

    /// The `with_scale=False` variant: no stored weight.
    pub fn without_scale(dim: usize, eps: f64, vb: &VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: Tensor::ones(dim, vb.dtype(), vb.device())?,
            eps: eps as f32,
        })
    }
}

impl Module for RmsNorm {
    /// Normalizes the last dim. Run it on a contiguous layout: on a transposed tensor candle falls
    /// back to strided kernels (38 of ~90 ms per vision layer on the M1 Pro, recipe step 2).
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::rms_norm(&x.contiguous()?, &self.weight, self.eps)
    }
}
