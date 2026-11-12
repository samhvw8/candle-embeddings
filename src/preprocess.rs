//! Image preprocessing ops shared by encoder families, and the [`Patches`] a ViT takes.
//!
//! The ops reproduce the reference processors bit for bit where it matters: the resize is
//! PyTorch's uint8 antialiased bicubic kernel (what torchvision runs), and the rescale is the
//! float32 product `x * (1/255)` that transformers computes.

use std::path::Path;

use anyhow::{Context, Result, bail};
use image::{DynamicImage, RgbImage};

/// A preprocessed image as `grid_h * grid_w` square patches, row-major, each flattened as
/// `(py, px, channel)`, values in `[0, 1]`.
#[derive(Debug, Clone)]
pub struct Patches {
    pub data: Vec<f32>,
    pub grid_w: usize,
    pub grid_h: usize,
    /// Patch side in pixels.
    pub patch: usize,
}

impl Patches {
    pub fn num_patches(&self) -> usize {
        self.grid_w * self.grid_h
    }

    /// Floats per patch: `3 * patch^2`.
    pub fn patch_dim(&self) -> usize {
        3 * self.patch * self.patch
    }

    /// Wrap patch data made elsewhere (a reference processor's output, in the exactness harness).
    pub fn from_raw(data: Vec<f32>, grid_w: usize, grid_h: usize, patch: usize) -> Result<Self> {
        let p = Self {
            data,
            grid_w,
            grid_h,
            patch,
        };
        if p.data.len() != p.num_patches() * p.patch_dim() {
            bail!(
                "patch data has {} floats, expected {grid_w}x{grid_h}x{}",
                p.data.len(),
                p.patch_dim()
            );
        }
        Ok(p)
    }
}

/// Open an image file and apply its EXIF orientation (transformers' `load_image` does the same).
pub fn open_image(path: &Path) -> Result<DynamicImage> {
    use image::ImageDecoder;
    let mut decoder = image::ImageReader::open(path)
        .with_context(|| format!("open {}", path.display()))?
        .with_guessed_format()?
        .into_decoder()
        .with_context(|| format!("decode {}", path.display()))?;
    let orientation = decoder.orientation()?;
    let mut img = DynamicImage::from_decoder(decoder)
        .with_context(|| format!("decode {}", path.display()))?;
    img.apply_orientation(orientation);
    Ok(img)
}

/// Bicubic resize with antialias on uint8, bit-exact with PyTorch's CPU kernel that torchvision
/// `resize(uint8, BICUBIC, antialias=True)` runs (`aten/src/ATen/native/cpu/UpSampleKernel.cpp`,
/// v2.11.0: `_compute_indices_min_size_weights_aa`, `_compute_index_ranges_int16_weights`,
/// `basic_loop_aa_horizontal<uint8_t>`). It follows Pillow's design: Keys cubic a = -0.5, the kernel
/// widened by the scale when shrinking, weights normalized per output pixel then quantized to int16,
/// width pass first, uint8 between passes.
pub fn resize_bicubic_aa(img: &RgbImage, out_w: usize, out_h: usize) -> RgbImage {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut buf = img.as_raw().clone();
    let mut cur_w = w;
    if out_w != w {
        let k = AxisKernel::new(w, out_w);
        let mut next = vec![0u8; out_w * h * 3];
        for y in 0..h {
            let src = &buf[y * w * 3..(y + 1) * w * 3];
            let dst = &mut next[y * out_w * 3..(y + 1) * out_w * 3];
            k.apply(dst, src, 3, 3);
        }
        buf = next;
        cur_w = out_w;
    }
    if out_h != h {
        let k = AxisKernel::new(h, out_h);
        let row = cur_w * 3;
        let mut next = vec![0u8; out_h * row];
        for x in 0..row {
            // One column of one channel: stride = row length.
            let col: Vec<u8> = (0..h).map(|y| buf[y * row + x]).collect();
            let mut out = vec![0u8; out_h];
            k.apply(&mut out, &col, 1, 1);
            for (y, v) in out.into_iter().enumerate() {
                next[y * row + x] = v;
            }
        }
        buf = next;
    }
    RgbImage::from_raw(out_w as u32, out_h as u32, buf).expect("buffer matches size")
}

/// Per-output-pixel source ranges and int16 weights along one axis.
struct AxisKernel {
    xmin: Vec<usize>,
    xsize: Vec<usize>,
    weights: Vec<i32>,
    stride: usize,
    precision: u32,
}

fn keys_cubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A
    } else {
        0.0
    }
}

impl AxisKernel {
    fn new(input: usize, output: usize) -> Self {
        let scale = input as f64 / output as f64;
        let support = if scale >= 1.0 { 2.0 * scale } else { 2.0 };
        let stride = support.ceil() as usize * 2 + 1;
        let invscale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
        let mut xmin = Vec::with_capacity(output);
        let mut xsize = Vec::with_capacity(output);
        let mut wf = vec![0f64; output * stride];
        let mut wt_max = 0f64;
        for i in 0..output {
            let center = scale * (i as f64 + 0.5);
            // C++ casts truncate toward zero.
            let lo = ((center - support + 0.5) as i64).max(0);
            let hi = ((center + support + 0.5) as i64).min(input as i64);
            let size = (hi - lo).clamp(0, stride as i64) as usize;
            let lo = lo as usize;
            let ws = &mut wf[i * stride..i * stride + size];
            let mut total = 0.0;
            for (j, w) in ws.iter_mut().enumerate() {
                *w = keys_cubic(((j + lo) as f64 - center + 0.5) * invscale);
                total += *w;
            }
            if total != 0.0 {
                for w in ws.iter_mut() {
                    *w /= total;
                    wt_max = wt_max.max(*w);
                }
            }
            xmin.push(lo);
            xsize.push(size);
        }
        let mut precision = 0u32;
        while precision < 22 {
            let next = (0.5 + wt_max * (1i64 << (precision + 1)) as f64) as i64;
            if next >= 1 << 15 {
                break;
            }
            precision += 1;
        }
        let weights = wf
            .iter()
            .map(|&w| {
                let v = w * (1i64 << precision) as f64;
                // int16 storage, rounded half away from zero.
                (if v < 0.0 {
                    (-0.5 + v) as i32
                } else {
                    (0.5 + v) as i32
                }) as i16 as i32
            })
            .collect();
        Self {
            xmin,
            xsize,
            weights,
            stride,
            precision,
        }
    }

    /// `src`/`dst` hold `channels` interleaved values per pixel.
    fn apply(&self, dst: &mut [u8], src: &[u8], channels: usize, step: usize) {
        let half = 1i32 << (self.precision - 1);
        for (i, (&lo, &size)) in self.xmin.iter().zip(&self.xsize).enumerate() {
            let ws = &self.weights[i * self.stride..i * self.stride + size];
            for c in 0..channels {
                let mut acc = half;
                for (j, &w) in ws.iter().enumerate() {
                    acc += src[(lo + j) * step + c] as i32 * w;
                }
                dst[i * step + c] = (acc >> self.precision).clamp(0, 255) as u8;
            }
        }
    }
}

/// Split an RGB image whose sides are multiples of `patch` into rescaled patches.
pub fn patchify(img: &RgbImage, patch: usize) -> Result<Patches> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    if patch == 0 || !w.is_multiple_of(patch) || !h.is_multiple_of(patch) {
        bail!("{w}x{h} image is not a whole number of {patch} px patches");
    }
    let (gw, gh) = (w / patch, h / patch);
    let raw = img.as_raw();
    let mut data = Vec::with_capacity(gw * gh * 3 * patch * patch);
    // transformers multiplies by rescale_factor = 1/255 (a float32 product in both backends).
    let scale = 1.0f32 / 255.0;
    for gy in 0..gh {
        for gx in 0..gw {
            for py in 0..patch {
                let row = (gy * patch + py) * w;
                for px in 0..patch {
                    let o = (row + gx * patch + px) * 3;
                    data.extend(raw[o..o + 3].iter().map(|&v| v as f32 * scale));
                }
            }
        }
    }
    Ok(Patches {
        data,
        grid_w: gw,
        grid_h: gh,
        patch,
    })
}

/// Largest per-pixel difference between two preprocessed images, in 1/255 units. `None` when their
/// patch grids differ.
pub fn pixel_maxdiff(a: &Patches, b: &Patches) -> Option<f32> {
    if (a.grid_w, a.grid_h, a.patch) != (b.grid_w, b.grid_h, b.patch) {
        return None;
    }
    Some(
        a.data
            .iter()
            .zip(&b.data)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
            * 255.0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patch_layout_is_row_major_py_px_channel() {
        let mut img = RgbImage::new(48, 48);
        for (x, y, p) in img.enumerate_pixels_mut() {
            *p = image::Rgb([x as u8, y as u8, 7]);
        }
        let p = patchify(&img, 16).unwrap();
        assert_eq!((p.grid_w, p.grid_h, p.patch_dim()), (3, 3, 768));
        // Patch 4 = grid (x 1, y 1); its pixel (py 2, px 5) = image (21, 18).
        let o = 4 * 768 + (2 * 16 + 5) * 3;
        let k = 1.0f32 / 255.0;
        assert_eq!(&p.data[o..o + 3], &[21.0 * k, 18.0 * k, 7.0 * k]);
        assert!(patchify(&RgbImage::new(40, 48), 16).is_err());
    }

    #[test]
    fn resize_keeps_a_flat_image_flat() {
        let img = RgbImage::from_pixel(97, 61, image::Rgb([200, 13, 77]));
        for (w, h) in [(48, 32), (240, 144), (97, 30)] {
            let out = resize_bicubic_aa(&img, w, h);
            assert_eq!((out.width(), out.height()), (w as u32, h as u32));
            assert!(out.pixels().all(|p| p.0 == [200, 13, 77]), "{w}x{h}");
        }
    }

    #[test]
    fn from_raw_checks_the_length() {
        assert!(Patches::from_raw(vec![0.0; 2 * 768], 2, 1, 16).is_ok());
        assert!(Patches::from_raw(vec![0.0; 767], 1, 1, 16).is_err());
    }
}
