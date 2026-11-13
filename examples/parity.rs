//! EmbeddingGemma 2 against the Python reference, with per-input lines, on any device / dtype / batch.
//!
//! cargo run --release -p candle-embeddings --example parity -- [metal|cuda|cpu] [f32|bf16|f16] [batch]

use std::path::Path;

use anyhow::{Result, bail};
use candle_core::{DType, Device};
use candle_embeddings::exact::{self, Spec};
use candle_embeddings::families::embeddinggemma2::{EmbeddingGemma2, MODEL_ID, prompts};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let device = match args.get(1).map(String::as_str).unwrap_or("metal") {
        "metal" => Device::new_metal(0)?,
        "cuda" => Device::new_cuda(0)?,
        "cpu" => Device::Cpu,
        d => bail!("unknown device {d}"),
    };
    let dtype = match args.get(2).map(String::as_str).unwrap_or("f32") {
        "f32" => DType::F32,
        "bf16" => DType::BF16,
        "f16" => DType::F16,
        d => bail!("unknown dtype {d}"),
    };
    let batch: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(4);
    let t = std::time::Instant::now();
    let emb = EmbeddingGemma2::load_with(MODEL_ID, &device, dtype)?.with_batch_size(batch);
    println!(
        "loaded {MODEL_ID} on {device:?} {dtype:?} in {:.1}s",
        t.elapsed().as_secs_f32()
    );
    let spec = Spec {
        fixtures: Path::new(env!("CARGO_MANIFEST_DIR")).join("bench/gemma"),
        text_prompts: &[
            ("clustering", Some(prompts::CLUSTERING)),
            ("query", Some(prompts::SEARCH_QUERY)),
        ],
        retrieval: "query",
    };
    let t = std::time::Instant::now();
    let rep = exact::run(&emb, &spec, true)?;
    println!("{}", rep.summary());
    println!("parity run {:.1}s", t.elapsed().as_secs_f32());
    Ok(())
}
