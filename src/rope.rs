//! Rotary position embeddings: 1-D (text) and 2-D axial (vision), both on candle's fused kernel.

use candle_core::{DType, Device, Result, Tensor};

/// cos and sin of `positions[p] * inv_freq[i]`, row-major (positions, inv_freq.len()), f32.
pub fn table(positions: impl Iterator<Item = u32>, inv_freq: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut cos = Vec::new();
    let mut sin = Vec::new();
    for p in positions {
        for &f in inv_freq {
            let a = p as f32 * f;
            cos.push(a.cos());
            sin.push(a.sin());
        }
    }
    (cos, sin)
}

/// Default RoPE frequencies: `1 / theta^(2i / head_dim)` for `i < head_dim / 2`.
pub fn default_inv_freq(head_dim: usize, theta: f64) -> Vec<f32> {
    (0..head_dim / 2)
        .map(|i| (1.0 / theta.powf((2 * i) as f64 / head_dim as f64)) as f32)
        .collect()
}

/// transformers `compute_axial_rope_parameters`: each spatial axis gets `head_dim / 2` dims, and
/// the frequencies run over `arange(0, head_dim / 2, 2)`.
pub fn axial_inv_freq(head_dim: usize, theta: f64) -> Vec<f32> {
    let spatial = head_dim / 2;
    (0..spatial)
        .step_by(2)
        .map(|i| (1.0 / theta.powf(i as f64 / spatial as f64)) as f32)
        .collect()
}

/// 1-D tables (s, head_dim / 2) for positions `0..s`, in `dtype`.
pub fn tables_1d(
    s: usize,
    head_dim: usize,
    theta: f64,
    dtype: DType,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let inv = default_inv_freq(head_dim, theta);
    let (cos, sin) = table(0..s as u32, &inv);
    let cos = Tensor::from_vec(cos, (s, head_dim / 2), device)?.to_dtype(dtype)?;
    let sin = Tensor::from_vec(sin, (s, head_dim / 2), device)?.to_dtype(dtype)?;
    Ok((cos, sin))
}

/// Per-axis 2-D tables (2, n, inv_freq.len()): axis 0 from `xs`, axis 1 from `ys`, in `dtype`.
pub fn tables_axial_2d(
    xs: &[u32],
    ys: &[u32],
    inv_freq: &[f32],
    dtype: DType,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let (n, f) = (xs.len(), inv_freq.len());
    let (cos, sin) = table(xs.iter().chain(ys).copied(), inv_freq);
    let cos = Tensor::from_vec(cos, (2, n, f), device)?.to_dtype(dtype)?;
    let sin = Tensor::from_vec(sin, (2, n, f), device)?.to_dtype(dtype)?;
    Ok((cos, sin))
}

/// Non-interleaved (`rotate_half`) RoPE on (b, h, s, d) with tables (s, d / 2).
pub fn apply(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    candle_nn::rotary_emb::rope(x, cos, sin)
}

/// `apply_multidimensional_rope` with 2 axes: the first half of each head rotates by the x
/// position, the second half by y, each half with its own `rotate_half`.
///
/// `x`: (b, s, h, hd); `cos`/`sin`: (2, s, hd / 4) from [`tables_axial_2d`]. Runs the fused 1-D kernel
/// by moving the axis to the front: (b, s, h, 2 * half) -> (2, b * h, s, half). Returns (b, h, s, hd),
/// contiguous.
pub fn apply_axial_2d(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b, s, h, hd) = x.dims4()?;
    let half = hd / 2;
    let x = x
        .reshape((b, s, h, 2, half))?
        .permute((3, 0, 2, 1, 4))?
        .contiguous()?
        .reshape((2, b * h, s, half))?;
    let y = candle_nn::rotary_emb::rope(&x, cos, sin)?;
    y.reshape((2, b, h, s, half))?
        .permute((1, 2, 3, 0, 4))?
        .contiguous()?
        .reshape((b, h, s, hd))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frequencies_match_transformers() {
        let f = default_inv_freq(8, 10000.0);
        assert_eq!(f.len(), 4);
        assert_eq!(f[0], 1.0);
        assert!((f[1] - 0.1).abs() < 1e-7);
        // head_dim 8: spatial 4, frequencies over i = 0, 2 -> theta^0, theta^-0.5.
        let a = axial_inv_freq(8, 100.0);
        assert_eq!(a, vec![1.0, 0.1]);
    }

    #[test]
    fn axial_rope_rotates_each_half_by_its_own_axis() -> Result<()> {
        // One head of 8 dims, one position at (x 1, y 0): the y half must stay unchanged.
        let dev = Device::Cpu;
        let inv = axial_inv_freq(8, 100.0);
        let (cos, sin) = tables_axial_2d(&[1], &[0], &inv, DType::F32, &dev)?;
        let x = Tensor::arange(1f32, 9.0, &dev)?.reshape((1, 1, 1, 8))?;
        let y: Vec<f32> = apply_axial_2d(&x, &cos, &sin)?.flatten_all()?.to_vec1()?;
        assert_eq!(&y[4..], &[5.0, 6.0, 7.0, 8.0]);
        // x half (1, 2 | 3, 4): rotate_half pairs (1, 3) by angle 1 and (2, 4) by angle 0.1.
        let (c, s) = (1f32.cos(), 1f32.sin());
        assert!((y[0] - (1.0 * c - 3.0 * s)).abs() < 1e-6);
        assert!((y[2] - (3.0 * c + 1.0 * s)).abs() < 1e-6);
        Ok(())
    }
}
