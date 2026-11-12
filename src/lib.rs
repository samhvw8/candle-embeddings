//! Embedding encoders in candle: shared blocks in the core, one module per model family.
//!
//! The core holds what every encoder needs: weight loading ([`weights`]), RMSNorm ([`norm`]),
//! RoPE ([`rope`]), attention with a per-backend fused path ([`attention`]), the gated MLP
//! ([`mlp`]), image preprocessing and patches ([`preprocess`]), pooling and L2 ([`pool`]), and the
//! exactness harness against a stored transformers reference ([`exact`]). A family under
//! [`families`] writes only its config, weight map and layer graph, and implements [`Encoder`].
//!
//! Every family follows `.okf/recipes/adding-an-encoder-family.md`: exact against its reference on
//! day one, then the speed steps in order. A family names each core block it does not use, and why.

pub mod attention;
pub mod exact;
pub mod families;
pub mod mlp;
pub mod norm;
pub mod pool;
pub mod preprocess;
pub mod rope;
pub mod weights;

use std::path::Path;

use anyhow::{Result, bail};
use candle_core::{DType, Device, Tensor};

pub use ::image::DynamicImage;
pub use preprocess::{Patches, open_image};

/// Last hidden states of one batch, before pooling.
pub struct Hidden {
    /// (b, s, dim) f32 on the model device. Rows are right-padded.
    pub states: Tensor,
    /// Valid length of each row.
    pub lengths: Vec<usize>,
}

/// How [`Encoder`] turns hidden states into one vector per input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
    /// Mean over every valid token, prompt included (sentence-transformers `pooling_mode_mean_tokens`,
    /// `include_prompt: true`).
    Mean,
}

/// One embedding model: preprocess -> forward -> pool -> L2.
///
/// A family implements the required methods. The provided `embed_*` methods split the input into
/// batches of [`Encoder::batch_size`], pool, normalize and reject non-finite output.
pub trait Encoder {
    /// A preprocessed image, ready for [`Encoder::forward_images`]. Made on the CPU, so a caller
    /// can prepare the next batch on another thread while the device runs this one.
    type Image;

    /// Output width.
    fn dim(&self) -> usize;
    fn device(&self) -> &Device;
    /// Compute dtype.
    fn dtype(&self) -> DType;
    /// Inputs per forward pass.
    fn batch_size(&self) -> usize;
    fn pooling(&self) -> Pooling;

    /// The family's image processor, on a decoded image (see [`open_image`]).
    fn preprocess(&self, img: &DynamicImage) -> Result<Self::Image>;
    /// One forward pass over at most `batch_size` images.
    fn forward_images(&self, batch: &[Self::Image], prompt: Option<&str>) -> Result<Hidden>;
    /// One forward pass over at most `batch_size` texts, each prefixed with `prompt`.
    fn forward_texts(&self, batch: &[&str], prompt: Option<&str>) -> Result<Hidden>;

    /// Pool and L2-normalize one batch. Fails on a non-finite value rather than return it.
    fn finish(&self, hidden: Hidden) -> Result<Vec<Vec<f32>>> {
        let pooled = match self.pooling() {
            Pooling::Mean => pool::mean(&hidden.states, &hidden.lengths)?,
        };
        let rows: Vec<Vec<f32>> = pool::l2_normalize(&pooled)?
            .to_device(&Device::Cpu)?
            .to_vec2()?;
        for r in &rows {
            if r.len() != self.dim() {
                bail!("embedding has {} dims, expected {}", r.len(), self.dim());
            }
            if r.iter().any(|x| !x.is_finite()) {
                bail!("non-finite embedding (dtype {:?})", self.dtype());
            }
        }
        Ok(rows)
    }

    /// Embed already-preprocessed images.
    fn embed_preprocessed(
        &self,
        images: &[Self::Image],
        prompt: Option<&str>,
    ) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(images.len());
        for chunk in images.chunks(self.batch_size().max(1)) {
            out.extend(self.finish(self.forward_images(chunk, prompt)?)?);
        }
        Ok(out)
    }

    /// Preprocess and embed decoded images.
    fn embed_images(&self, images: &[DynamicImage], prompt: Option<&str>) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(images.len());
        for chunk in images.chunks(self.batch_size().max(1)) {
            let pre = chunk
                .iter()
                .map(|i| self.preprocess(i))
                .collect::<Result<Vec<_>>>()?;
            out.extend(self.embed_preprocessed(&pre, prompt)?);
        }
        Ok(out)
    }

    /// Open, orient (EXIF), preprocess and embed image files.
    fn embed_image_paths<P: AsRef<Path>>(
        &self,
        paths: &[P],
        prompt: Option<&str>,
    ) -> Result<Vec<Vec<f32>>>
    where
        Self: Sized,
    {
        let mut out = Vec::with_capacity(paths.len());
        for chunk in paths.chunks(self.batch_size().max(1)) {
            let pre = chunk
                .iter()
                .map(|p| self.preprocess(&open_image(p.as_ref())?))
                .collect::<Result<Vec<_>>>()?;
            out.extend(self.embed_preprocessed(&pre, prompt)?);
        }
        Ok(out)
    }

    /// Embed texts, each prefixed with `prompt`.
    fn embed_texts(&self, texts: &[&str], prompt: Option<&str>) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(self.batch_size().max(1)) {
            out.extend(self.finish(self.forward_texts(chunk, prompt)?)?);
        }
        Ok(out)
    }
}
