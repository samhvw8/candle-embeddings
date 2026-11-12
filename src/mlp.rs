//! Activations and the gated MLP.

use candle_core::{Module, Result, Tensor};
use candle_nn::Linear;

/// Activation functions, by their transformers config name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// `gelu_pytorch_tanh`.
    GeluTanh,
}

impl Activation {
    /// Parse a transformers `hidden_activation` / `hidden_act` value. Refuses names not built here.
    pub fn from_config(name: &str) -> anyhow::Result<Self> {
        match name {
            "gelu_pytorch_tanh" => Ok(Self::GeluTanh),
            other => anyhow::bail!("activation {other:?} is not implemented"),
        }
    }

    pub fn apply(self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::GeluTanh => x.gelu(),
        }
    }
}

/// `down(act(gate(x)) * up(x))`. The family loads the three linears from its own weight names.
#[derive(Debug, Clone)]
pub struct GatedMlp {
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
    pub act: Activation,
}

impl Module for GatedMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let g = self.act.apply(&self.gate.forward(xs)?)?;
        self.down.forward(&(g * self.up.forward(xs)?)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_activation_is_refused() {
        assert_eq!(
            Activation::from_config("gelu_pytorch_tanh").unwrap(),
            Activation::GeluTanh
        );
        assert!(Activation::from_config("silu").is_err());
    }
}
