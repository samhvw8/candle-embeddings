//! Weights: the memory-mapped checkpoint, token-table rows read from the map, linear layers, and
//! model directories in the Hugging Face cache.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device};
use candle_nn::{Linear, VarBuilder};
use safetensors::Dtype;

/// One safetensors checkpoint file, memory-mapped on demand.
pub struct Checkpoint {
    path: PathBuf,
}

impl Checkpoint {
    pub fn new(path: &Path) -> Result<Self> {
        ensure!(path.is_file(), "no checkpoint at {}", path.display());
        Ok(Self {
            path: path.to_path_buf(),
        })
    }

    /// A var builder over the mapped file. Tensors are read only when a layer asks for them, so a
    /// family loads just its towers (EmbeddingGemma 2 never reads the audio tower).
    pub fn var_builder(&self, dtype: DType, device: &Device) -> Result<VarBuilder<'static>> {
        // SAFETY: the file is not modified while mapped (a read-only HF cache blob or a local copy).
        Ok(unsafe { VarBuilder::from_mmaped_safetensors(&[&self.path], dtype, device)? })
    }

    /// A [`TokenTable`] over the tensor `name`, of shape (vocab, hidden).
    pub fn token_table(&self, name: &str, vocab: usize, hidden: usize) -> Result<TokenTable> {
        // SAFETY: as in `var_builder`.
        let st = unsafe { MmapedSafetensors::new(&self.path)? };
        TokenTable::new(st, name, vocab, hidden)
    }
}

/// A linear layer without bias, weight at `<vb>.weight` of shape (out, in).
pub fn linear_no_bias(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilder,
) -> candle_core::Result<Linear> {
    candle_nn::linear_no_bias(in_dim, out_dim, vb)
}

/// Rows of a token embedding table, read straight from the memory-mapped checkpoint. A large
/// vocabulary (EmbeddingGemma 2: 262144 x 512, 268 MB in bf16) needs a few hundred rows a batch, so
/// the table is never loaded whole.
pub struct TokenTable {
    st: MmapedSafetensors,
    name: String,
    vocab: usize,
    hidden: usize,
    dtype: Dtype,
}

impl TokenTable {
    pub fn new(st: MmapedSafetensors, name: &str, vocab: usize, hidden: usize) -> Result<Self> {
        let view = st.get(name)?;
        if view.shape() != [vocab, hidden] {
            bail!(
                "{name}: shape {:?}, expected [{vocab}, {hidden}]",
                view.shape()
            );
        }
        let dtype = view.dtype();
        if !matches!(dtype, Dtype::BF16 | Dtype::F16 | Dtype::F32) {
            bail!("{name}: unsupported dtype {dtype:?}");
        }
        Ok(Self {
            st,
            name: name.to_string(),
            vocab,
            hidden,
            dtype,
        })
    }

    /// Embedding rows as f32, `ids.len() * hidden`.
    pub fn rows(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let view = self.st.get(&self.name)?;
        let data = view.data();
        let width = match self.dtype {
            Dtype::F32 => 4,
            _ => 2,
        };
        let row_bytes = self.hidden * width;
        let mut out = Vec::with_capacity(ids.len() * self.hidden);
        for &id in ids {
            let id = id as usize;
            if id >= self.vocab {
                bail!("token id {id} out of range (vocab {})", self.vocab);
            }
            let row = &data[id * row_bytes..(id + 1) * row_bytes];
            match self.dtype {
                Dtype::BF16 => out.extend(
                    row.as_chunks::<2>()
                        .0
                        .iter()
                        .map(|b| half::bf16::from_le_bytes(*b).to_f32()),
                ),
                Dtype::F16 => out.extend(
                    row.as_chunks::<2>()
                        .0
                        .iter()
                        .map(|b| half::f16::from_le_bytes(*b).to_f32()),
                ),
                _ => out.extend(
                    row.as_chunks::<4>()
                        .0
                        .iter()
                        .map(|b| f32::from_le_bytes(*b)),
                ),
            }
        }
        Ok(out)
    }
}

/// A local directory that holds `files`, or a Hugging Face id (`owner/name`) resolved through the
/// local Hugging Face cache at `revision` (a branch, tag or commit hash).
///
/// The cache is read directly, in the layout `huggingface_hub` writes:
/// `<hub cache>/models--{owner}--{name}/refs/{revision}` holds a commit hash, and the files are in
/// `snapshots/{hash}/`. The hub cache is `HF_HUB_CACHE`, else `HUGGINGFACE_HUB_CACHE`, else
/// `<HF_HOME>/hub`, where `HF_HOME` defaults to `$XDG_CACHE_HOME/huggingface`, else
/// `~/.cache/huggingface`.
///
/// On a cache miss, `download` fetches the files when the crate is built with the `download`
/// feature (hf-hub); without the feature, or with `download` false, a miss is an error that says
/// how to fetch the model (`hf download {model} --revision {revision}`).
pub fn model_dir(model: &str, revision: &str, files: &[&str], download: bool) -> Result<PathBuf> {
    let p = Path::new(model);
    if p.is_dir() {
        for f in files {
            ensure!(p.join(f).is_file(), "{} has no {f}", p.display());
        }
        return Ok(p.to_path_buf());
    }
    let (owner, name) = model
        .split_once('/')
        .with_context(|| format!("{model:?} is neither a directory nor an owner/name HF id"))?;
    let miss = match cached_snapshot(owner, name, revision, files) {
        Ok(dir) => return Ok(dir),
        Err(e) => e,
    };
    if download {
        return fetch(model, revision, files).with_context(|| format!("{miss:#}"));
    }
    Err(miss)
}

/// The hub cache directory, resolved like `huggingface_hub` and `hf-hub`.
pub fn hub_cache_dir() -> Result<PathBuf> {
    for var in ["HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"] {
        if let Some(v) = std::env::var_os(var).filter(|v| !v.is_empty()) {
            return Ok(PathBuf::from(v));
        }
    }
    let home = if let Some(v) = std::env::var_os("HF_HOME").filter(|v| !v.is_empty()) {
        PathBuf::from(v)
    } else if let Some(v) = std::env::var_os("XDG_CACHE_HOME").filter(|v| !v.is_empty()) {
        PathBuf::from(v).join("huggingface")
    } else {
        let h = std::env::var_os("HOME")
            .filter(|v| !v.is_empty())
            .context("no HOME, HF_HOME or HF_HUB_CACHE to find the Hugging Face cache")?;
        PathBuf::from(h).join(".cache").join("huggingface")
    };
    Ok(home.join("hub"))
}

/// `snapshots/{hash}/` of `owner/name` at `revision` in the local cache, when it holds every file.
fn cached_snapshot(owner: &str, name: &str, revision: &str, files: &[&str]) -> Result<PathBuf> {
    let model = format!("{owner}/{name}");
    let hint = format!(
        "fetch it with `hf download {model} --revision {revision}`, or build encoder with the `download` feature"
    );
    let repo = hub_cache_dir()?.join(format!("models--{owner}--{name}"));
    ensure!(
        repo.is_dir(),
        "{model} is not in the Hugging Face cache ({} is missing); {hint}",
        repo.display()
    );
    let refs = repo.join("refs").join(revision);
    let hash = if refs.is_file() {
        std::fs::read_to_string(&refs)
            .with_context(|| format!("read {}", refs.display()))?
            .trim()
            .to_string()
    } else {
        // A commit hash has no ref file; its snapshot folder is named after it.
        revision.to_string()
    };
    let snap = repo.join("snapshots").join(&hash);
    ensure!(
        snap.is_dir(),
        "{model}@{revision} is not in the Hugging Face cache ({} is missing); {hint}",
        snap.display()
    );
    for f in files {
        ensure!(
            snap.join(f).is_file(),
            "{model}@{revision}: {f} is not in the Hugging Face cache ({}); {hint}",
            snap.display()
        );
    }
    Ok(snap)
}

/// Download `files` of `model` at `revision` into the HF cache (feature `download`).
#[cfg(feature = "download")]
fn fetch(model: &str, revision: &str, files: &[&str]) -> Result<PathBuf> {
    let (owner, name) = model.split_once('/').context("owner/name HF id")?;
    let client = hf_hub::HFClientSync::new().context("hf-hub client")?;
    let repo = client.model(owner, name);
    let mut dir = None;
    for f in files {
        let path = repo
            .download_file()
            .filename(*f)
            .revision(revision)
            .send()
            .with_context(|| format!("download {model}/{f}@{revision}"))?;
        dir = path.parent().map(Path::to_path_buf);
    }
    dir.context("no model files resolved")
}

#[cfg(not(feature = "download"))]
fn fetch(model: &str, revision: &str, _files: &[&str]) -> Result<PathBuf> {
    bail!(
        "{model}@{revision} is not cached and this build has no `download` feature; run `hf download {model} --revision {revision}`"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fake cache: `refs/main` -> hash -> `snapshots/<hash>/` with the files.
    #[test]
    fn resolves_a_ref_and_a_hash_from_the_cache_layout() {
        let root = std::env::temp_dir().join(format!("encoder-hfcache-{}", std::process::id()));
        let repo = root.join("models--acme--tiny");
        let hash = "0123456789abcdef0123456789abcdef01234567";
        std::fs::create_dir_all(repo.join("refs")).unwrap();
        std::fs::create_dir_all(repo.join("snapshots").join(hash)).unwrap();
        std::fs::write(repo.join("refs/main"), format!("{hash}\n")).unwrap();
        std::fs::write(repo.join("snapshots").join(hash).join("config.json"), "{}").unwrap();
        // SAFETY: tests in this module are the only readers of HF_HUB_CACHE here.
        unsafe { std::env::set_var("HF_HUB_CACHE", &root) };
        let want = repo.join("snapshots").join(hash);
        assert_eq!(
            model_dir("acme/tiny", "main", &["config.json"], false).unwrap(),
            want
        );
        assert_eq!(
            model_dir("acme/tiny", hash, &["config.json"], false).unwrap(),
            want
        );
        let e = model_dir("acme/tiny", "main", &["missing.bin"], false).unwrap_err();
        assert!(format!("{e:#}").contains("hf download acme/tiny"), "{e:#}");
        let e = model_dir("acme/other", "main", &["config.json"], false).unwrap_err();
        assert!(
            format!("{e:#}").contains("not in the Hugging Face cache"),
            "{e:#}"
        );
        unsafe { std::env::remove_var("HF_HUB_CACHE") };
        std::fs::remove_dir_all(&root).unwrap();
    }
}
