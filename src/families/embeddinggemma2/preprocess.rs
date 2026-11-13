//! Image preprocessing: transformers `Gemma4ImageProcessor` (5.22.2) on the core ops.
//!
//! 1. Convert to RGB (alpha dropped, as PIL `convert("RGB")` does).
//! 2. Resize with the core's uint8 bicubic + antialias, keeping the aspect ratio, to the largest size
//!    whose sides are multiples of `patch * pooling` (48 px) and that yields at most
//!    `max_soft_tokens * pooling^2` patches (`get_aspect_ratio_preserving_size`, the part that is
//!    this family's own). `max_soft_tokens` is the vision token budget: 280 by default
//!    (`processor_config.json`), or another of [`SUPPORTED_SOFT_TOKENS`].
//! 3. Rescale to `[0, 1]` (no mean/std normalization; the vision tower maps to `[-1, 1]` itself).
//! 4. Cut into 16x16 patches, row-major, each flattened as `(py, px, channel)`.

use anyhow::{Result, bail};
use image::DynamicImage;

use crate::preprocess::{Patches, patchify, resize_bicubic_aa};

pub const PATCH_SIZE: usize = 16;
pub const POOLING_KERNEL: usize = 3;
/// The model's default vision token budget (`image_processor.max_soft_tokens` in `processor_config.json`).
pub const MAX_SOFT_TOKENS: usize = 280;
/// Budgets `Gemma4ImageProcessor` accepts (`_SUPPORTED_SOFT_TOKENS`, transformers 5.22.2). The model
/// card: "soft token amounts ranging from 70 to 1120"; fewer tokens trade quality for speed.
pub const SUPPORTED_SOFT_TOKENS: [usize; 5] = [70, 140, 280, 560, 1120];
pub const PATCH_DIM: usize = 3 * PATCH_SIZE * PATCH_SIZE;

/// Soft tokens the vision tower yields for an image (one per 3x3 patch block). Fails on a patch grid
/// this model cannot take.
pub fn soft_tokens(p: &Patches) -> Result<usize> {
    if p.patch != PATCH_SIZE
        || !p.grid_w.is_multiple_of(POOLING_KERNEL)
        || !p.grid_h.is_multiple_of(POOLING_KERNEL)
    {
        bail!(
            "{}x{} grid of {} px patches: expected {PATCH_SIZE} px patches and sides that are multiples of {POOLING_KERNEL}",
            p.grid_w,
            p.grid_h,
            p.patch
        );
    }
    Ok(p.num_patches() / (POOLING_KERNEL * POOLING_KERNEL))
}

/// Fails on a budget the image processor rejects.
pub fn check_budget(max_soft_tokens: usize) -> Result<()> {
    if !SUPPORTED_SOFT_TOKENS.contains(&max_soft_tokens) {
        bail!("max_soft_tokens must be one of {SUPPORTED_SOFT_TOKENS:?}, got {max_soft_tokens}");
    }
    Ok(())
}

/// `get_aspect_ratio_preserving_size` from transformers. Returns `(height, width)` in pixels.
pub fn target_size(height: usize, width: usize, max_soft_tokens: usize) -> Result<(usize, usize)> {
    let max_patches = max_soft_tokens * POOLING_KERNEL * POOLING_KERNEL;
    let total_px = (height * width) as f64;
    let target_px = (max_patches * PATCH_SIZE * PATCH_SIZE) as f64;
    let factor = (target_px / total_px).sqrt();
    let ideal_h = factor * height as f64;
    let ideal_w = factor * width as f64;
    let side = POOLING_KERNEL * PATCH_SIZE;
    let mut th = (ideal_h / side as f64).floor() as usize * side;
    let mut tw = (ideal_w / side as f64).floor() as usize * side;
    if th == 0 && tw == 0 {
        bail!("cannot resize {height}x{width} to a 0x0 image");
    }
    let max_side = (max_patches / (POOLING_KERNEL * POOLING_KERNEL)) * side;
    if th == 0 {
        th = side;
        tw = ((width as f64 / height as f64).floor() as usize * side).min(max_side);
    } else if tw == 0 {
        tw = side;
        th = ((height as f64 / width as f64).floor() as usize * side).min(max_side);
    }
    if (th * tw) as f64 > target_px {
        bail!("resizing {height}x{width} to {th}x{tw} exceeds {max_patches} patches");
    }
    Ok((th, tw))
}

/// Preprocess one image with the model's default budget of 280 soft tokens.
pub fn preprocess(img: &DynamicImage) -> Result<Patches> {
    preprocess_with(img, MAX_SOFT_TOKENS)
}

/// Preprocess one image with a vision token budget from [`SUPPORTED_SOFT_TOKENS`].
pub fn preprocess_with(img: &DynamicImage, max_soft_tokens: usize) -> Result<Patches> {
    check_budget(max_soft_tokens)?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    if w == 0 || h == 0 {
        bail!("empty image");
    }
    let (th, tw) = target_size(h, w, max_soft_tokens)?;
    let resized = if (th, tw) == (h, w) {
        rgb
    } else {
        resize_bicubic_aa(&rgb, tw, th)
    };
    patchify(&resized, PATCH_SIZE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_sizes_match_reference() {
        // (w, h) -> grid (w, h), from bench/gemma/ref/images.json (transformers 5.22.2).
        let cases = [
            ((512, 512), (48, 48)),
            ((451, 300), (60, 39)),
            ((1000, 872), (51, 45)),
            ((384, 191), (69, 33)),
            ((600, 150), (99, 24)),
            ((380, 30), (177, 12)),
            ((90, 1200), (12, 183)),
            ((64, 48), (57, 42)),
        ];
        for ((w, h), (gw, gh)) in cases {
            let (th, tw) = target_size(h, w, MAX_SOFT_TOKENS).unwrap();
            assert_eq!((tw / PATCH_SIZE, th / PATCH_SIZE), (gw, gh), "{w}x{h}");
        }
    }

    #[test]
    fn smaller_budgets_match_reference() {
        // (w, h) -> grid (w, h) at 70 and 140 soft tokens, from `reference.py --max-soft-tokens N`.
        let cases = [
            ((512, 512), (24, 24), (33, 33)),
            ((451, 300), (30, 18), (42, 27)),
            ((1000, 872), (24, 21), (36, 33)),
            ((384, 191), (33, 15), (48, 24)),
            ((600, 150), (48, 12), (69, 15)),
            ((380, 30), (87, 6), (126, 9)),
            ((90, 1200), (6, 90), (9, 129)),
            ((64, 48), (27, 21), (39, 30)),
        ];
        for ((w, h), g70, g140) in cases {
            for (budget, (gw, gh)) in [(70, g70), (140, g140)] {
                let (th, tw) = target_size(h, w, budget).unwrap();
                assert_eq!(
                    (tw / PATCH_SIZE, th / PATCH_SIZE),
                    (gw, gh),
                    "{w}x{h} @ {budget}"
                );
            }
        }
    }

    #[test]
    fn budget_must_be_a_supported_one() {
        let img = DynamicImage::new_rgb8(96, 96);
        for b in SUPPORTED_SOFT_TOKENS {
            preprocess_with(&img, b).unwrap();
        }
        assert!(preprocess_with(&img, 100).is_err());
        assert!(preprocess_with(&img, 0).is_err());
    }

    #[test]
    fn extreme_aspect_uses_the_fallback_branch() {
        // 1 x 4000 rounds the height to zero, so the height becomes one 48 px row.
        let (th, tw) = target_size(1, 4000, MAX_SOFT_TOKENS).unwrap();
        assert_eq!(th, 48);
        assert_eq!(tw, 280 * 48);
        assert!(target_size(0, 0, MAX_SOFT_TOKENS).is_err());
    }

    #[test]
    fn soft_tokens_need_a_grid_of_threes() {
        let p = Patches::from_raw(vec![0.0; 6 * 3 * PATCH_DIM], 6, 3, PATCH_SIZE).unwrap();
        assert_eq!(soft_tokens(&p).unwrap(), 2);
        let q = Patches::from_raw(vec![0.0; 4 * 3 * PATCH_DIM], 4, 3, PATCH_SIZE).unwrap();
        assert!(soft_tokens(&q).is_err());
    }
}
