//! Attention: the head-layout helper, plain attention with a mask, the fused per-backend path, and
//! the bidirectional padding / sliding-window mask.

use candle_core::{D, DType, Device, Result, Tensor};

/// (b, s, heads, d) -> (b, heads, s, d), contiguous with canonical strides.
///
/// candle 0.11.1's Metal matmul miscomputes when a size-1 dim keeps a non-canonical stride, which
/// `transpose(1, 2).contiguous()` leaves on a one-kv-head tensor (it counts as contiguous, so no copy
/// is made). The final reshape rebuilds the layout from the shape. Recipe step 1.
pub fn heads_first(x: &Tensor) -> Result<Tensor> {
    let (b, s, h, d) = x.dims4()?;
    x.transpose(1, 2)?.contiguous()?.reshape((b, h, s, d))
}

/// Which kernel [`attention_fused`] runs for these inputs on this device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Path {
    /// candle's fused Metal kernel (`candle_nn::ops::sdpa`): never builds the (s x s) scores.
    MetalSdpa,
    /// FlashAttention-2 (`candle-flash-attn`) on CUDA, bf16/f16 only, head_dim a multiple of 8 up
    /// to 256. Built with the `flash-attn` feature. Recipe step 5.
    #[cfg(feature = "flash-attn")]
    CudaFlash,
    /// [`attention`]: the score matrix in memory, softmax in f32.
    Plain,
}

/// The kernel [`attention_fused`] picks.
pub fn fused_path(q: &Tensor) -> Result<Path> {
    let hd = q.dim(D::Minus1)?;
    let metal_hd =
        matches!(hd, 32 | 64 | 72 | 80 | 96 | 128 | 256) || (hd == 512 && q.dtype() != DType::F32);
    if q.device().is_metal() && metal_hd {
        return Ok(Path::MetalSdpa);
    }
    // f32 stays on the plain path: it is the exact one (FlashAttention runs only f16/bf16).
    #[cfg(feature = "flash-attn")]
    if q.device().is_cuda()
        && matches!(q.dtype(), DType::BF16 | DType::F16)
        && hd.is_multiple_of(8)
        && hd <= 256
    {
        return Ok(Path::CudaFlash);
    }
    Ok(Path::Plain)
}

/// Unmasked attention, scale 1.0, through the fastest kernel the backend has (see [`fused_path`]).
/// A fused kernel saves the score matrix: 12 x 2304^2 f32 scores are 255 MB per image in a ViT.
/// Falls back to [`attention`].
///
/// `q`: (b, h, s, d); `k`, `v`: (b, h_kv, s, d). Returns (b, h, s, d); on CUDA flash it is a
/// transposed view of a contiguous (b, s, h, d), so the caller's `transpose(1, 2)` back to
/// (b, s, h, d) costs no copy.
pub fn attention_fused(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    match fused_path(q)? {
        Path::MetalSdpa => candle_nn::ops::sdpa(q, k, v, None, false, 1.0, 1.0),
        #[cfg(feature = "flash-attn")]
        Path::CudaFlash => {
            // The kernel takes (b, s, heads, d) with any strides but a contiguous last dim.
            let t = |x: &Tensor| x.transpose(1, 2);
            candle_flash_attn::flash_attn(&t(q)?, &t(k)?, &t(v)?, 1.0, false)?.transpose(1, 2)
        }
        Path::Plain => attention(q, k, v, None),
    }
}

/// softmax(q k^T) v with scale 1.0 (families that normalize q and k instead of scaling, or that
/// fold the scale into q), softmax in f32.
///
/// `q`: (b, h, s, d); `k`, `v`: (b, h_kv, s, d) with `h` a multiple of `h_kv` (GQA); `mask`: additive
/// f32 (b, s, s).
pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
    let (b, h, s, d) = q.dims4()?;
    let h_kv = k.dim(1)?;
    let groups = h / h_kv;
    // Fold the query-head groups into the sequence axis so k/v need no copy for GQA.
    let q = q.reshape((b, h_kv, groups * s, d))?;
    let scores = q.matmul(&k.t()?)?;
    let mut scores = scores.to_dtype(DType::F32)?;
    if let Some(m) = mask {
        scores = scores
            .reshape((b, h_kv, groups, s, s))?
            .broadcast_add(&m.unsqueeze(1)?.unsqueeze(1)?)?
            .reshape((b, h_kv, groups * s, s))?;
    }
    let probs = candle_nn::ops::softmax_last_dim(&scores)?.to_dtype(v.dtype())?;
    probs.matmul(v)?.reshape((b, h, s, d))
}

/// Additive f32 mask (b, s, s) for bidirectional attention over right-padded rows: keys at or past
/// a row's length are blocked, and with `window`, keys farther than `window` from a valid query.
///
/// Padding queries (at or past the length) get no window: a padding query far past the end would
/// otherwise have every key blocked, its softmax would be NaN, and the NaN would reach valid rows
/// through `0 * NaN` in the next layer's matmuls. Their outputs never reach a valid row (valid rows
/// block padded keys) nor the pooled mean (weight 0).
pub fn padding_mask(
    lengths: &[usize],
    s: usize,
    window: Option<usize>,
    device: &Device,
) -> Result<Tensor> {
    let mut m = vec![0f32; lengths.len() * s * s];
    for (bi, &len) in lengths.iter().enumerate() {
        for q in 0..s {
            for k in 0..s {
                let blocked = k >= len || (q < len && window.is_some_and(|w| q.abs_diff(k) > w));
                if blocked {
                    m[(bi * s + q) * s + k] = f32::NEG_INFINITY;
                }
            }
        }
    }
    Tensor::from_vec(m, (lengths.len(), s, s), device)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FlashAttention on CUDA in bf16 (GQA, ViT-like shape) against the plain path in f32 on the
    /// same bf16 inputs: at least as close as the plain bf16 path, which rounds its scores to bf16
    /// before the softmax. Skips without a GPU.
    #[cfg(feature = "flash-attn")]
    #[test]
    fn flash_is_as_close_to_f32_as_plain_bf16() -> Result<()> {
        let Ok(dev) = Device::new_cuda(0) else {
            eprintln!("skip: no CUDA device");
            return Ok(());
        };
        let r = |h| Tensor::randn(0f32, 1., (2, h, 300, 64), &dev)?.to_dtype(DType::BF16);
        let (q, k, v) = (r(12)?, r(4)?, r(4)?);
        assert_eq!(fused_path(&q)?, Path::CudaFlash);
        assert_eq!(fused_path(&q.to_dtype(DType::F32)?)?, Path::Plain);
        let f = |t: &Tensor| t.to_dtype(DType::F32);
        let want = attention(&f(&q)?, &f(&k)?, &f(&v)?, None)?;
        let err = |t: Tensor| -> Result<f32> { (f(&t)? - &want)?.abs()?.max_all()?.to_scalar() };
        let flash = err(attention_fused(&q, &k, &v)?)?;
        let plain = err(attention(&q, &k, &v, None)?)?;
        eprintln!("max |diff| vs f32: flash {flash}, plain bf16 {plain}");
        assert!(flash <= plain, "flash {flash} > plain bf16 {plain}");
        Ok(())
    }

    /// A padded query row more than `window` past its row's end must keep a key: a fully blocked
    /// row gives NaN, and NaN values leak into valid rows through `0 * NaN` in the next matmul.
    #[test]
    fn no_query_row_is_fully_blocked() -> Result<()> {
        let m: Vec<Vec<Vec<f32>>> = padding_mask(&[2, 10], 10, Some(3), &Device::Cpu)?.to_vec3()?;
        for (b, rows) in m.iter().enumerate() {
            for (q, row) in rows.iter().enumerate() {
                assert!(row.contains(&0.0), "batch {b} query {q} fully blocked");
            }
        }
        // Valid queries keep the window.
        assert_eq!(m[1][9][5], f32::NEG_INFINITY);
        assert_eq!(m[1][9][6], 0.0);
        Ok(())
    }

    #[test]
    fn mask_blocks_padding_and_the_window() -> Result<()> {
        let m: Vec<Vec<Vec<f32>>> = padding_mask(&[3, 2], 3, Some(1), &Device::Cpu)?.to_vec3()?;
        let inf = f32::NEG_INFINITY;
        assert_eq!(
            m[0],
            vec![vec![0., 0., inf], vec![0., 0., 0.], vec![inf, 0., 0.]]
        );
        // Query 2 of row 1 is padding: no window, padded key still blocked.
        assert_eq!(m[1][2], vec![0., 0., inf]);
        Ok(())
    }

    #[test]
    fn gqa_attention_matches_repeated_kv() -> Result<()> {
        let dev = Device::Cpu;
        let q = Tensor::randn(0f32, 1., (1, 4, 5, 8), &dev)?;
        let k = Tensor::randn(0f32, 1., (1, 2, 5, 8), &dev)?;
        let v = Tensor::randn(0f32, 1., (1, 2, 5, 8), &dev)?;
        let got = attention(&q, &k, &v, None)?;
        // Reference: repeat each kv head for its 2 query heads, then plain softmax(q k^T) v.
        let rep = |t: &Tensor| -> Result<Tensor> {
            Tensor::cat(
                &[
                    t.narrow(1, 0, 1)?,
                    t.narrow(1, 0, 1)?,
                    t.narrow(1, 1, 1)?,
                    t.narrow(1, 1, 1)?,
                ],
                1,
            )
        };
        let (k4, v4) = (rep(&k)?, rep(&v)?);
        let p = candle_nn::ops::softmax_last_dim(&q.matmul(&k4.t()?)?)?;
        let want = p.matmul(&v4)?;
        let diff: f32 = (got - want)?.abs()?.max_all()?.to_scalar()?;
        assert!(diff < 1e-5, "{diff}");
        assert_eq!(fused_path(&q)?, Path::Plain);
        Ok(())
    }
}
