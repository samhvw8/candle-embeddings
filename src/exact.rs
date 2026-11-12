//! Exactness harness: a family against vectors stored from its transformers reference.
//!
//! The fixtures directory is what a family's reference script writes (template:
//! `bench/gemma/reference.py`):
//!
//! | file | content |
//! |---|---|
//! | `images/<name>.png` (path in `file`) | public test images |
//! | `ref/images.json` | `[{name, file, grid: [w, h], input_ids, caption}]` |
//! | `ref/images.f32` | n x dim reference image vectors, raw LE f32 |
//! | `ref/pixels/<name>.f32` | the reference processor's patches, for the model-only check |
//! | `ref/texts.json` | `{texts: [...]}` |
//! | `ref/texts_<key>.f32` | reference text vectors with prompt `<key>` |
//!
//! Checks: token ids of each image input, pixels after preprocessing (max diff in 1/255), cosine of
//! every vector (model only with the reference pixels, end to end with ours, texts), and image ->
//! caption top-1 retrieval compared with the reference vectors' own score.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use serde::Deserialize;

use crate::preprocess::{Patches, pixel_maxdiff};
use crate::{Encoder, open_image};

/// A family the harness can check: images are [`Patches`].
pub trait ExactCheck: Encoder<Image = Patches> {
    /// Patch side of the reference processor's output.
    fn patch_size(&self) -> usize;
    /// Token ids of one image input, compared with the reference processor's `input_ids`. `None`
    /// for a family whose image input has no token sequence.
    fn image_input_ids(&self, _img: &Patches) -> Result<Option<Vec<u32>>> {
        Ok(None)
    }
}

#[derive(Debug, Deserialize)]
pub struct RefImage {
    pub name: String,
    pub file: String,
    pub grid: [usize; 2],
    pub input_ids: Vec<u32>,
    pub caption: String,
}

#[derive(Debug, Deserialize)]
pub struct RefTexts {
    pub texts: Vec<String>,
}

/// What to check.
pub struct Spec<'a> {
    /// Directory with `ref/` and the image files.
    pub fixtures: PathBuf,
    /// `(key, prompt)`: `ref/texts_<key>.f32` holds the reference vectors for that prompt.
    pub text_prompts: &'a [(&'a str, Option<&'a str>)],
    /// The text key whose vectors the image -> caption retrieval uses.
    pub retrieval: &'a str,
}

impl Spec<'_> {
    pub fn available(&self) -> bool {
        self.fixtures.join("ref/images.json").is_file()
    }
}

pub fn read_f32(path: &Path) -> Result<Vec<f32>> {
    let b = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let (words, rest) = b.as_chunks::<4>();
    ensure!(
        rest.is_empty(),
        "{}: length not a multiple of 4",
        path.display()
    );
    Ok(words.iter().map(|c| f32::from_le_bytes(*c)).collect())
}

/// Cosine in f64.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        d += (*x as f64) * (*y as f64);
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    (d / (na.sqrt() * nb.sqrt())) as f32
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

#[derive(Debug, Default)]
pub struct Report {
    /// Model only: reference pixels in, cosine vs reference vector.
    pub image_model: Vec<(String, f32)>,
    /// End to end: our preprocessing, cosine vs reference vector.
    pub image_e2e: Vec<(String, f32)>,
    /// Max |pixel| difference of our preprocessing vs the reference processor, in 1/255 units.
    pub pixel_maxdiff: Vec<(String, f32)>,
    pub token_ids_match: bool,
    pub text: Vec<(String, f32)>,
    /// Largest |element| difference of any vector vs its reference.
    pub max_abs_diff: f32,
    /// Images whose top caption is their own, with our vectors.
    pub retrieval_hits: usize,
    /// The same count from the reference vectors.
    pub ref_retrieval_hits: usize,
    pub retrieval_total: usize,
}

/// Smallest cosine an f32 path may have against the f32 reference (measured: 1.000000).
pub const F32_MIN_COSINE: f32 = 0.99999;
/// Largest element difference an f32 path may have (measured: 1.23e-6).
pub const F32_MAX_ABS_DIFF: f32 = 1e-4;

fn stats(v: &[(String, f32)]) -> (f32, f32) {
    let min = v.iter().map(|x| x.1).fold(f32::INFINITY, f32::min);
    let mean = v.iter().map(|x| x.1).sum::<f32>() / v.len().max(1) as f32;
    (min, mean)
}

impl Report {
    pub fn summary(&self) -> String {
        let (a, b) = stats(&self.image_model);
        let (c, d) = stats(&self.image_e2e);
        let (e, f) = stats(&self.text);
        let (_, pmean) = stats(&self.pixel_maxdiff);
        let pmax = self.pixel_maxdiff.iter().map(|x| x.1).fold(0f32, f32::max);
        format!(
            "image model-only cos min {a:.6} mean {b:.6} | image e2e cos min {c:.6} mean {d:.6} | \
             text cos min {e:.6} mean {f:.6} | max |diff| {:.2e} | pixel maxdiff max {pmax:.1} mean {pmean:.2} /255 | \
             token ids match {} | retrieval {}/{} (reference {}/{})",
            self.max_abs_diff,
            self.token_ids_match,
            self.retrieval_hits,
            self.retrieval_total,
            self.ref_retrieval_hits,
            self.retrieval_total
        )
    }

    pub fn min_cosine(&self) -> f32 {
        let all: Vec<_> = self
            .image_model
            .iter()
            .chain(&self.image_e2e)
            .chain(&self.text)
            .cloned()
            .collect();
        stats(&all).0
    }

    /// The gate of an exact (f32) path: [`Report::check`] at [`F32_MIN_COSINE`]
    /// and no element further than [`F32_MAX_ABS_DIFF`] from its reference. A
    /// half-precision path fails it (bf16 scores 0.999784), so a lossy path
    /// left on by accident cannot pass as exact.
    pub fn check_f32(&self) -> Result<()> {
        self.check(F32_MIN_COSINE)?;
        ensure!(
            self.max_abs_diff <= F32_MAX_ABS_DIFF,
            "max element difference {} is above {F32_MAX_ABS_DIFF}: {}",
            self.max_abs_diff,
            self.summary()
        );
        Ok(())
    }

    /// The stage-0 gate: token ids equal, pixels equal, every cosine at least `min_cos`, retrieval
    /// equal to the reference's.
    pub fn check(&self, min_cos: f32) -> Result<()> {
        ensure!(self.token_ids_match, "token ids differ");
        // An empty fixture set or a NaN must fail: `f32::min` drops NaN, and
        // a gate over nothing passes.
        ensure!(
            !self.image_model.is_empty() && !self.image_e2e.is_empty() && !self.text.is_empty(),
            "nothing compared: {} images (model), {} images (end to end), {} texts",
            self.image_model.len(),
            self.image_e2e.len(),
            self.text.len()
        );
        let cosines = self
            .image_model
            .iter()
            .chain(&self.image_e2e)
            .chain(&self.text);
        if let Some((name, c)) = cosines.into_iter().find(|(_, c)| !c.is_finite()) {
            anyhow::bail!("{name}: cosine is {c}");
        }
        ensure!(
            self.max_abs_diff.is_finite(),
            "element difference is {}",
            self.max_abs_diff
        );
        ensure!(
            self.pixel_maxdiff.iter().all(|(_, d)| *d == 0.0),
            "pixels differ: {:?}",
            self.pixel_maxdiff
        );
        ensure!(self.min_cosine() >= min_cos, "{}", self.summary());
        ensure!(
            self.retrieval_hits == self.ref_retrieval_hits,
            "{}",
            self.summary()
        );
        Ok(())
    }
}

/// Run every check. `verbose` prints per-input lines.
pub fn run<E: ExactCheck>(enc: &E, spec: &Spec, verbose: bool) -> Result<Report> {
    let dir = &spec.fixtures;
    let dim = enc.dim();
    let refs: Vec<RefImage> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("ref/images.json"))?)?;
    let ref_img = read_f32(&dir.join("ref/images.f32"))?;
    ensure!(ref_img.len() == refs.len() * dim, "ref/images.f32 size");
    let mut rep = Report {
        token_ids_match: true,
        ..Default::default()
    };

    let mut model_in = Vec::new();
    let mut e2e_in = Vec::new();
    for r in &refs {
        let px = read_f32(&dir.join(format!("ref/pixels/{}.f32", r.name)))?;
        let p = Patches::from_raw(px, r.grid[0], r.grid[1], enc.patch_size())?;
        if let Some(ids) = enc.image_input_ids(&p)?
            && ids != r.input_ids
        {
            rep.token_ids_match = false;
            eprintln!(
                "{}: token ids differ ({} vs {})",
                r.name,
                ids.len(),
                r.input_ids.len()
            );
        }
        let mine = enc.preprocess(&open_image(&dir.join(&r.file))?)?;
        let diff = pixel_maxdiff(&mine, &p).unwrap_or(f32::INFINITY);
        rep.pixel_maxdiff.push((r.name.clone(), diff));
        model_in.push(p);
        e2e_in.push(mine);
    }
    let model_vecs = enc.embed_preprocessed(&model_in, None)?;
    let e2e_vecs = enc.embed_preprocessed(&e2e_in, None)?;
    for (i, r) in refs.iter().enumerate() {
        let want = &ref_img[i * dim..(i + 1) * dim];
        let cm = cosine(&model_vecs[i], want);
        let ce = cosine(&e2e_vecs[i], want);
        rep.max_abs_diff = rep
            .max_abs_diff
            .max(max_abs(&model_vecs[i], want))
            .max(max_abs(&e2e_vecs[i], want));
        if verbose {
            println!(
                "  {:12} grid {:3}x{:3}  model-only {cm:.6}  e2e {ce:.6}  pixel maxdiff {:.1}",
                r.name, r.grid[0], r.grid[1], rep.pixel_maxdiff[i].1
            );
        }
        rep.image_model.push((r.name.clone(), cm));
        rep.image_e2e.push((r.name.clone(), ce));
    }

    let texts: RefTexts =
        serde_json::from_str(&std::fs::read_to_string(dir.join("ref/texts.json"))?)?;
    let tx: Vec<&str> = texts.texts.iter().map(String::as_str).collect();
    let mut retrieval_vecs = None;
    for &(key, prompt) in spec.text_prompts {
        let want = read_f32(&dir.join(format!("ref/texts_{key}.f32")))?;
        ensure!(want.len() == tx.len() * dim, "ref/texts_{key}.f32 size");
        let got = enc.embed_texts(&tx, prompt)?;
        for (i, g) in got.iter().enumerate() {
            let w = &want[i * dim..(i + 1) * dim];
            let c = cosine(g, w);
            rep.max_abs_diff = rep.max_abs_diff.max(max_abs(g, w));
            if verbose && (i == 0 || i + 1 == got.len() || c < 0.9999) {
                println!("  text[{key}] {i:2} ({} chars) cos {c:.6}", tx[i].len());
            }
            rep.text.push((format!("{key}:{i}"), c));
        }
        if key == spec.retrieval {
            retrieval_vecs = Some((got, want));
        }
    }
    let (query_vecs, ref_query) = retrieval_vecs
        .with_context(|| format!("retrieval key {:?} is not in text_prompts", spec.retrieval))?;
    let ref_t = |j: usize| &ref_query[j * dim..(j + 1) * dim];
    let top = |score: &dyn Fn(usize) -> f32| {
        (0..tx.len())
            .max_by(|&a, &b| score(a).total_cmp(&score(b)))
            .expect("texts")
    };
    for (i, r) in refs.iter().enumerate() {
        let ref_v = &ref_img[i * dim..(i + 1) * dim];
        let ref_best = top(&|j| cosine(ref_v, ref_t(j)));
        rep.ref_retrieval_hits += (tx[ref_best] == r.caption) as usize;
        let best = top(&|j| cosine(&e2e_vecs[i], &query_vecs[j]));
        let hit = tx[best] == r.caption;
        if verbose {
            println!(
                "  retrieval {:12} -> {}{}",
                r.name,
                &tx[best][..tx[best].len().min(50)],
                if hit { "" } else { "  (miss)" }
            );
        }
        rep.retrieval_hits += hit as usize;
        rep.retrieval_total += 1;
    }
    Ok(rep)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn passing() -> Report {
        let one = |n: &str| vec![(n.to_string(), 1.0f32)];
        Report {
            image_model: one("a"),
            image_e2e: one("a"),
            pixel_maxdiff: vec![("a".into(), 0.0)],
            token_ids_match: true,
            text: one("t"),
            ..Default::default()
        }
    }

    #[test]
    fn check_fails_on_nan_and_on_nothing_compared() {
        passing().check(0.999).unwrap();
        // A NaN cosine (a zero or NaN vector) is not a pass.
        let mut r = passing();
        r.text.push(("bad".into(), f32::NAN));
        assert!(r.check(0.999).is_err(), "NaN cosine passed the gate");
        // Empty fixtures compare nothing.
        let r = Report {
            token_ids_match: true,
            ..Default::default()
        };
        assert!(r.check(0.999).is_err(), "an empty report passed the gate");
        let mut r = passing();
        r.text[0].1 = 0.99;
        assert!(r.check(0.999).is_err());
    }

    #[test]
    fn f32_gate_rejects_a_half_precision_result() {
        passing().check_f32().unwrap();
        // The measured bf16 result: it passes the loose gate, not the f32 gate.
        let mut r = passing();
        r.text[0].1 = 0.999_784;
        r.check(0.999).unwrap();
        assert!(r.check_f32().is_err(), "a bf16 cosine passed the f32 gate");
        // Cosine fine, one element far off.
        let mut r = passing();
        r.max_abs_diff = 4e-3;
        assert!(r.check_f32().is_err(), "a large element difference passed");
    }
}
