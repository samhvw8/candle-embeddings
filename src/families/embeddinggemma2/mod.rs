//! EmbeddingGemma 2 (`google/embeddinggemma-2`) image and text embeddings.
//!
//! An image becomes the token sequence `<bos> <|image> <|image|> x N <image|> <eos>`, where the N
//! placeholder rows (N = soft tokens, at most 280) are replaced by the vision tower's output. Text is
//! `<bos> prompt + text <eos>`. The text encoder runs bidirectionally over the sequence and projects
//! every token to 768-d; the core then takes the mean over all tokens and L2-normalizes — the
//! sentence-transformers pipeline (Transformer -> Pooling(mean) -> Normalize) of the model repo.
//!
//! The model card says to pass images without a task prefix, so `None` is the prompt for photos.
//! Text prompts are in [`prompts`].
//!
//! Own code: config, weight map (`Gemma4ClippableLinear` names), the input sequence, the Gemma 4
//! vision tower and the EmbeddingGemma 2 text encoder (PLE, per-layer attention shapes), and the
//! aspect-ratio target size. Core blocks it skips: none.

pub mod config;
pub mod preprocess;
mod text;
mod vision;

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor};
use image::DynamicImage;
use tokenizers::Tokenizer;

use crate::exact::ExactCheck;
use crate::mlp::Activation;
use crate::preprocess::Patches;
use crate::weights::{Checkpoint, TokenTable, model_dir};
use crate::{Encoder, Hidden, Pooling};

pub use config::Config;
pub use preprocess::{PATCH_DIM, PATCH_SIZE, preprocess, soft_tokens};

/// Output width.
pub const EMBED_DIM: usize = 768;
/// Hugging Face repo of the model.
pub const MODEL_ID: &str = "google/embeddinggemma-2";
/// Pinned revision: the snapshot the parity numbers were measured on.
pub const MODEL_REVISION: &str = "914f7f89142e33e77833254d9c9b90c3cef7303b";
/// Context length of the text encoder (model card).
pub const MAX_TOKENS: usize = 8192;
/// Images (or texts) per forward pass. On the M1 Pro, batches of 1, 4 and 8 images ran at the same
/// rate and memory grew with the batch (recipe, step 4), so the default is 1.
pub const DEFAULT_BATCH_SIZE: usize = 1;

/// Task prefixes from `config_sentence_transformers.json`. Images take none.
pub mod prompts {
    pub const CLUSTERING: &str = "task: clustering | query: ";
    pub const CLASSIFICATION: &str = "task: classification | query: ";
    pub const SEARCH_QUERY: &str = "task: search result | query: ";
    pub const DOCUMENT: &str = "title: none | text: ";
    pub const SENTENCE_SIMILARITY: &str = "task: sentence similarity | query: ";
}

const IMAGE_TOKEN: &str = "<|image|>";
const BOI_TOKEN: &str = "<|image>";
const EOI_TOKEN: &str = "<image|>";
const MODEL_FILES: [&str; 3] = ["config.json", "tokenizer.json", "model.safetensors"];

/// What one sequence of a batch holds.
struct Item {
    ids: Vec<u32>,
    /// Index into the batch's soft-token list, and where the placeholders start.
    image: Option<(usize, usize)>,
}

pub struct EmbeddingGemma2 {
    cfg: Config,
    tokenizer: Tokenizer,
    table: TokenTable,
    vision: vision::VisionTower,
    text: text::TextEncoder,
    batch_size: usize,
}

impl EmbeddingGemma2 {
    /// Load from a local model directory or a Hugging Face id, in f32: exact parity with the
    /// reference (dtype table in the recipe).
    pub fn load(model: &str, device: &Device) -> Result<Self> {
        Self::load_with(model, device, DType::F32)
    }

    /// Load with an explicit compute dtype (F32 or BF16; F16 overflows, per the model card).
    pub fn load_with(model: &str, device: &Device, dtype: DType) -> Result<Self> {
        ensure!(
            matches!(dtype, DType::F32 | DType::BF16 | DType::F16),
            "unsupported dtype {dtype:?}"
        );
        let dir = resolve_model_dir(model)?;
        let cfg = Config::from_file(&dir.join("config.json"))?;
        let mut tokenizer =
            Tokenizer::from_file(dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: MAX_TOKENS,
                ..Default::default()
            }))
            .map_err(anyhow::Error::msg)?;
        for (tok, id) in [
            (IMAGE_TOKEN, cfg.image_token_id),
            (BOI_TOKEN, cfg.boi_token_id),
            (EOI_TOKEN, cfg.eoi_token_id),
        ] {
            ensure!(
                tokenizer.token_to_id(tok) == Some(id),
                "tokenizer maps {tok} to {:?}, config says {id}",
                tokenizer.token_to_id(tok)
            );
        }
        let ckpt = Checkpoint::new(&dir.join("model.safetensors"))?;
        let vb = ckpt.var_builder(dtype, device)?;
        let text = text::TextEncoder::new(
            &cfg.text_config,
            Activation::from_config(&cfg.text_config.hidden_activation)?,
            vb.pp("language_model"),
        )?;
        let vision = vision::VisionTower::new(
            &cfg.vision_config,
            Activation::from_config(&cfg.vision_config.hidden_activation)?,
            cfg.text_config.hidden_size,
            vb.pp("vision_tower"),
            vb.pp("embed_vision"),
        )?;
        drop(vb);
        let table = ckpt.token_table(
            "language_model.embed_tokens.weight",
            cfg.text_config.vocab_size,
            cfg.text_config.hidden_size,
        )?;
        Ok(Self {
            cfg,
            tokenizer,
            table,
            vision,
            text,
            batch_size: DEFAULT_BATCH_SIZE,
        })
    }

    /// Items per forward pass (images or texts). Larger batches use more memory.
    pub fn with_batch_size(mut self, n: usize) -> Self {
        self.batch_size = n.max(1);
        self
    }

    /// The token ids of one image input (exposed for parity tests).
    pub fn image_token_ids(
        &self,
        num_soft_tokens: usize,
        prompt: Option<&str>,
    ) -> Result<Vec<u32>> {
        Ok(self.image_item(0, num_soft_tokens, prompt)?.ids)
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .tokenizer
            .encode(text, true)
            .map_err(anyhow::Error::msg)?;
        Ok(enc.get_ids().to_vec())
    }

    /// `EmbeddingGemma2Processor`: the `<|image|>` placeholder expands to boi + N image tokens + eoi.
    fn image_item(&self, index: usize, n: usize, prompt: Option<&str>) -> Result<Item> {
        let text = format!(
            "{}{BOI_TOKEN}{}{EOI_TOKEN}",
            prompt.unwrap_or(""),
            IMAGE_TOKEN.repeat(n)
        );
        let ids = self.tokenize(&text)?;
        let img = self.cfg.image_token_id;
        let start = ids
            .iter()
            .position(|&t| t == img)
            .context("no image token after tokenization")?;
        let count = ids.iter().filter(|&&t| t == img).count();
        ensure!(
            count == n && ids[start..start + n].iter().all(|&t| t == img),
            "image placeholders: expected {n} contiguous, found {count} (prompt too long?)"
        );
        Ok(Item {
            ids,
            image: Some((index, start)),
        })
    }

    /// Soft tokens per image, (n_i, text_hidden), grouping images that share a patch grid.
    fn encode_images(&self, patches: &[Patches]) -> Result<Vec<Tensor>> {
        let dev = self.device();
        let mut out: Vec<Option<Tensor>> = vec![None; patches.len()];
        let mut done = vec![false; patches.len()];
        for i in 0..patches.len() {
            if done[i] {
                continue;
            }
            let grid = (patches[i].grid_w, patches[i].grid_h);
            let group: Vec<usize> = (i..patches.len())
                .filter(|&j| !done[j] && (patches[j].grid_w, patches[j].grid_h) == grid)
                .collect();
            let n = patches[i].num_patches();
            let mut data = Vec::with_capacity(group.len() * n * PATCH_DIM);
            for &j in &group {
                data.extend_from_slice(&patches[j].data);
                done[j] = true;
            }
            let px = Tensor::from_vec(data, (group.len(), n, PATCH_DIM), dev)?;
            let soft = self.vision.forward(&px, grid.0, grid.1)?;
            for (gi, &j) in group.iter().enumerate() {
                out[j] = Some(soft.get(gi)?);
            }
        }
        Ok(out
            .into_iter()
            .map(|t| t.expect("every image encoded"))
            .collect())
    }

    /// Build the padded input embeddings for a batch and run the text encoder.
    fn run(&self, items: &[Item], soft: &[Tensor]) -> Result<Hidden> {
        let dev = self.device();
        let hidden = self.cfg.text_config.hidden_size;
        let s = items
            .iter()
            .map(|it| it.ids.len())
            .max()
            .context("empty batch")?;
        // EmbeddingGemma2TextScaledWordEmbedding: rows * sqrt(hidden). Placeholder rows are
        // replaced below, so their ids do not matter (transformers looks up the pad id).
        let scale = (hidden as f32).sqrt();
        let mut rows = Vec::with_capacity(items.len());
        let mut lengths = Vec::with_capacity(items.len());
        for it in items {
            let ids: Vec<u32> = it
                .ids
                .iter()
                .map(|&t| {
                    if t == self.cfg.image_token_id {
                        self.cfg.text_config.pad_token_id
                    } else {
                        t
                    }
                })
                .collect();
            let mut e = self.table.rows(&ids)?;
            e.iter_mut().for_each(|v| *v *= scale);
            let len = ids.len();
            let mut t = Tensor::from_vec(e, (len, hidden), dev)?.to_dtype(self.dtype())?;
            if let Some((idx, start)) = it.image {
                let st = &soft[idx];
                let n = st.dim(0)?;
                let mut parts = vec![t.narrow(0, 0, start)?, st.to_dtype(self.dtype())?];
                if start + n < len {
                    parts.push(t.narrow(0, start + n, len - start - n)?);
                }
                t = Tensor::cat(&parts, 0)?;
            }
            if len < s {
                t = Tensor::cat(
                    &[t, Tensor::zeros((s - len, hidden), self.dtype(), dev)?],
                    0,
                )?;
            }
            rows.push(t);
            lengths.push(len);
        }
        let embeds = Tensor::stack(&rows, 0)?;
        let states = self.text.forward(&embeds, &lengths)?;
        Ok(Hidden { states, lengths })
    }
}

impl Encoder for EmbeddingGemma2 {
    type Image = Patches;

    fn dim(&self) -> usize {
        EMBED_DIM
    }

    fn device(&self) -> &Device {
        self.text.device()
    }

    fn dtype(&self) -> DType {
        self.text.dtype()
    }

    fn batch_size(&self) -> usize {
        self.batch_size
    }

    fn pooling(&self) -> Pooling {
        Pooling::Mean
    }

    fn preprocess(&self, img: &DynamicImage) -> Result<Patches> {
        preprocess(img)
    }

    fn forward_images(&self, batch: &[Patches], prompt: Option<&str>) -> Result<Hidden> {
        let items = batch
            .iter()
            .enumerate()
            .map(|(i, p)| self.image_item(i, soft_tokens(p)?, prompt))
            .collect::<Result<Vec<_>>>()?;
        let soft = self.encode_images(batch)?;
        self.run(&items, &soft)
    }

    fn forward_texts(&self, batch: &[&str], prompt: Option<&str>) -> Result<Hidden> {
        let p = prompt.unwrap_or("");
        let items = batch
            .iter()
            .map(|t| {
                Ok(Item {
                    ids: self.tokenize(&format!("{p}{t}"))?,
                    image: None,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        self.run(&items, &[])
    }
}

impl ExactCheck for EmbeddingGemma2 {
    fn patch_size(&self) -> usize {
        PATCH_SIZE
    }

    fn image_input_ids(&self, img: &Patches) -> Result<Option<Vec<u32>>> {
        Ok(Some(self.image_token_ids(soft_tokens(img)?, None)?))
    }
}

/// A local directory with the model files, or a HF id resolved through the local HF cache (the
/// pinned revision for [`MODEL_ID`], `main` otherwise). A cache miss downloads with the `download`
/// feature and is an error without it.
pub fn resolve_model_dir(model: &str) -> Result<std::path::PathBuf> {
    model_dir(model, revision(model), &MODEL_FILES, true)
}

/// Like [`resolve_model_dir`] but never downloads: `None` when the files are not cached.
pub fn cached_model_dir(model: &str) -> Option<std::path::PathBuf> {
    model_dir(model, revision(model), &MODEL_FILES, false).ok()
}

fn revision(model: &str) -> &'static str {
    if model == MODEL_ID {
        MODEL_REVISION
    } else {
        "main"
    }
}
