//! EmbeddingGemma 2 against the Python reference, with per-input lines, on any device / dtype / batch.
//!
//! cargo run --release -p candle-embeddings --example parity -- [metal|cuda|cpu] [f32|bf16|f16] [batch] [soft tokens]
//!
//! A vision token budget other than 280 reads `bench/gemma/soft<N>` (written by
//! `reference.py --max-soft-tokens N`) and also prints each image's cosine to the 280-token reference.

use std::path::Path;

use anyhow::{Result, bail};
use candle_core::{DType, Device};
use candle_embeddings::Encoder;
use candle_embeddings::exact::{self, RefImage, Spec};
use candle_embeddings::families::embeddinggemma2::{
    EMBED_DIM, EmbeddingGemma2, MAX_SOFT_TOKENS, MODEL_ID, prompts,
};

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
    let budget: usize = args
        .get(4)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(MAX_SOFT_TOKENS);
    let t = std::time::Instant::now();
    let emb = EmbeddingGemma2::load_with(MODEL_ID, &device, dtype)?
        .with_batch_size(batch)
        .with_max_soft_tokens(budget)?;
    println!(
        "loaded {MODEL_ID} on {device:?} {dtype:?} in {:.1}s, {budget} soft tokens",
        t.elapsed().as_secs_f32()
    );
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("bench/gemma");
    let spec = Spec {
        fixtures: if budget == MAX_SOFT_TOKENS {
            base.clone()
        } else {
            base.join(format!("soft{budget}"))
        },
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
    if budget != MAX_SOFT_TOKENS {
        // Our vectors at this budget against the 280-token reference (which we match at 1.000000).
        let refs: Vec<RefImage> =
            serde_json::from_str(&std::fs::read_to_string(base.join("ref/images.json"))?)?;
        let want = exact::read_f32(&base.join("ref/images.f32"))?;
        let paths: Vec<_> = refs.iter().map(|r| base.join(&r.file)).collect();
        let got = emb.embed_image_paths(&paths, None)?;
        let mut cos: Vec<f32> = Vec::new();
        for (i, r) in refs.iter().enumerate() {
            let c = exact::cosine(&got[i], &want[i * EMBED_DIM..(i + 1) * EMBED_DIM]);
            println!("  vs 280: {:12} {c:.6}", r.name);
            cos.push(c);
        }
        cos.sort_by(f32::total_cmp);
        println!(
            "cosine vs 280 soft tokens: min {:.6} median {:.6}",
            cos[0],
            cos[cos.len() / 2]
        );
    }
    Ok(())
}
