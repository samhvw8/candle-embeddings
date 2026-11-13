//! EmbeddingGemma 2 against the official sentence-transformers path (bench/gemma/reference.py, CPU
//! f32), through the core exactness harness.
//!
//! `smoke` runs in `cargo test` when the model is in the HF cache and the reference files exist.
//! Otherwise it skips and prints a `SKIPPED` line straight to stderr, which the test harness does
//! not capture. `full_parity` is the stage-0 gate and is ignored; run it with
//! `cargo test --release -p candle-embeddings --test embeddinggemma2 -- --ignored`.
//!
//! Both run on Metal on macOS and fail when Metal is missing: a CPU run would not test the device
//! production uses. `EMEANS_ENCODER_DEVICE=cpu` asks for the CPU on purpose.

use std::path::Path;

use candle_core::{DType, Device};
use candle_embeddings::Encoder;
use candle_embeddings::exact::{self, RefImage, Spec};
use candle_embeddings::families::embeddinggemma2::{
    EMBED_DIM, EmbeddingGemma2, MODEL_ID, cached_model_dir, prompts,
};

const TEXT_PROMPTS: &[(&str, Option<&str>)] = &[
    ("clustering", Some(prompts::CLUSTERING)),
    ("query", Some(prompts::SEARCH_QUERY)),
];

fn spec() -> Spec<'static> {
    Spec {
        fixtures: Path::new(env!("CARGO_MANIFEST_DIR")).join("bench/gemma"),
        text_prompts: TEXT_PROMPTS,
        retrieval: "query",
    }
}

/// Metal on macOS (a failure to open it is an error, never a silent CPU run);
/// the CPU elsewhere, or when `EMEANS_ENCODER_DEVICE=cpu` asks for it.
fn device() -> Device {
    let asked = std::env::var("EMEANS_ENCODER_DEVICE").ok();
    match asked.as_deref() {
        Some("cpu") => Device::Cpu,
        Some("metal") | None if cfg!(target_os = "macos") => Device::new_metal(0).expect(
            "Metal was expected but is not available; set EMEANS_ENCODER_DEVICE=cpu to test on the CPU",
        ),
        Some("metal") => panic!("EMEANS_ENCODER_DEVICE=metal needs macOS"),
        None => Device::Cpu,
        Some(other) => panic!("EMEANS_ENCODER_DEVICE must be cpu or metal, got {other}"),
    }
}

/// Say that a test did not run. Written to stderr directly: the test harness
/// captures `eprintln!` of a passing test, so that would hide the skip.
fn skipped(why: &str) {
    use std::io::Write;
    let _ = writeln!(std::io::stderr(), "\nSKIPPED candle-embeddings test: {why}");
}

fn load(dtype: DType) -> Option<EmbeddingGemma2> {
    let Some(dir) = cached_model_dir(MODEL_ID) else {
        skipped(&format!("{MODEL_ID} is not in the HF cache"));
        return None;
    };
    if !spec().available() {
        skipped("no reference outputs (run bench/gemma/reference.py)");
        return None;
    }
    let device = device();
    eprintln!("device: {device:?}");
    Some(EmbeddingGemma2::load_with(dir.to_str().unwrap(), &device, dtype).unwrap())
}

#[test]
fn smoke() {
    let Some(emb) = load(DType::F32) else { return };
    let dir = spec().fixtures;
    let refs: Vec<RefImage> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("ref/images.json")).unwrap())
            .unwrap();
    let want = exact::read_f32(&dir.join("ref/images.f32")).unwrap();
    // The 4:1 crop and the upscaled thumbnail: both resize directions, a wide grid.
    for name in ["coffee_wide", "cat_small"] {
        let i = refs.iter().position(|r| r.name == name).unwrap();
        let got = emb
            .embed_image_paths(&[dir.join(&refs[i].file)], None)
            .unwrap()
            .remove(0);
        let norm: f32 = got.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "{name}: norm {norm}");
        let c = exact::cosine(&got, &want[i * EMBED_DIM..(i + 1) * EMBED_DIM]);
        assert!(c >= exact::F32_MIN_COSINE, "{name}: cosine {c}");
        assert_eq!(
            emb.image_token_ids(refs[i].input_ids.len() - 4, None)
                .unwrap(),
            refs[i].input_ids
        );
    }
}

#[test]
#[ignore = "loads the model and embeds every reference input twice (~1 min on Metal)"]
fn full_parity() {
    let Some(emb) = load(DType::F32) else {
        panic!("full_parity needs the cached model and bench/gemma/ref");
    };
    // Batch 1, then batch 8: a batch mixes text lengths (one text is 1238 tokens), which caught a
    // fully masked padding row that batch 1 never builds.
    let rep = exact::run(&emb, &spec(), false).unwrap();
    println!("batch 1: {}", rep.summary());
    rep.check_f32().unwrap();
    let emb = emb.with_batch_size(8);
    let rep = exact::run(&emb, &spec(), false).unwrap();
    println!("batch 8: {}", rep.summary());
    rep.check_f32().unwrap();
}
