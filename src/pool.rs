//! Pooling: sequence mean, L2, and the spatial k x k sum of a ViT patch grid.

use candle_core::{Result, Tensor};

/// Mean over each row's first `lengths[i]` positions: (b, s, d) f32 -> (b, d).
///
/// One (b, 1, s) x (b, s, d) matmul with weights `1 / len` on the valid positions and 0 on padding.
pub fn mean(states: &Tensor, lengths: &[usize]) -> Result<Tensor> {
    let (b, s, _) = states.dims3()?;
    let mut w = vec![0f32; b * s];
    for (bi, &len) in lengths.iter().enumerate() {
        w[bi * s..bi * s + len].fill(1.0 / len as f32);
    }
    let w = Tensor::from_vec(w, (b, 1, s), states.device())?;
    w.matmul(states)?.squeeze(1)
}

/// Divide each row of (b, d) by its L2 norm.
pub fn l2_normalize(x: &Tensor) -> Result<Tensor> {
    let norm = x.sqr()?.sum_keepdim(1)?.sqrt()?;
    x.broadcast_div(&norm)
}

/// Sum of each `k x k` block of a row-major patch grid: (b, grid_h * grid_w, d) ->
/// (b, (grid_h / k) * (grid_w / k), d), in the input dtype.
///
/// It returns the sum, not the mean: fold `1 / k^2` into the scale that follows, so the result is
/// rounded once, as the reference does (Gemma 4: `* sqrt(hidden) / k^2` in one f32 product).
pub fn grid_sum(h: &Tensor, grid_w: usize, grid_h: usize, k: usize) -> Result<Tensor> {
    let (b, n, d) = h.dims3()?;
    if n != grid_w * grid_h || !grid_w.is_multiple_of(k) || !grid_h.is_multiple_of(k) {
        candle_core::bail!("grid_sum: {n} rows for a {grid_w}x{grid_h} grid with k = {k}");
    }
    let (gw, gh) = (grid_w / k, grid_h / k);
    h.reshape((b, gh, k, gw, k, d))?
        .sum(4)?
        .sum(2)?
        .reshape((b, gh * gw, d))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn mean_ignores_padding() -> Result<()> {
        let x = Tensor::new(&[[[1f32, 2.], [3., 4.], [100., 100.]]], &Device::Cpu)?;
        let m: Vec<Vec<f32>> = mean(&x, &[2])?.to_vec2()?;
        assert_eq!(m, vec![vec![2., 3.]]);
        let n: Vec<Vec<f32>> =
            l2_normalize(&Tensor::new(&[[3f32, 4.]], &Device::Cpu)?)?.to_vec2()?;
        assert_eq!(n, vec![vec![0.6, 0.8]]);
        Ok(())
    }

    #[test]
    fn grid_sum_adds_each_block() -> Result<()> {
        // 4 x 2 grid (w 4, h 2), k 2: blocks are columns {0,1} and {2,3} of both rows.
        let x = Tensor::arange(0f32, 8., &Device::Cpu)?.reshape((1, 8, 1))?;
        let s: Vec<f32> = grid_sum(&x, 4, 2, 2)?.flatten_all()?.to_vec1()?;
        assert_eq!(s, vec![0. + 1. + 4. + 5., 2. + 3. + 6. + 7.]);
        assert!(grid_sum(&x, 3, 2, 2).is_err());
        Ok(())
    }
}
