//! Images per second for one device / dtype / batch size, on the bench images cycled to `n`.
//!
//! /usr/bin/time -l target/.../examples/throughput metal f32 4 60   (metal|cuda|cpu)
//!
//! Prints the rate for embedding from files (decode + preprocess + model) after one warm-up batch.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Result, bail};
use candle_core::{DType, Device};
use candle_embeddings::Encoder;
use candle_embeddings::families::embeddinggemma2::{EmbeddingGemma2, MODEL_ID, preprocess};
use candle_embeddings::open_image;

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
    let n: usize = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(60);

    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("bench/gemma/images");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "png"))
        .collect();
    files.sort();
    if files.is_empty() {
        bail!(
            "no images in {} (run bench/gemma/reference.py)",
            dir.display()
        );
    }
    let paths: Vec<PathBuf> = files.iter().cycle().take(n).cloned().collect();

    let t = Instant::now();
    let emb = EmbeddingGemma2::load_with(MODEL_ID, &device, dtype)?.with_batch_size(batch);
    let load = t.elapsed().as_secs_f64();
    emb.embed_image_paths(&paths[..batch.min(n)], None)?;

    let t = Instant::now();
    let pre: Vec<_> = paths
        .iter()
        .map(|p| preprocess(&open_image(p)?))
        .collect::<Result<_>>()?;
    let pre_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let out = emb.embed_preprocessed(&pre, None)?;
    let model_s = t.elapsed().as_secs_f64();
    assert_eq!(out.len(), n);
    println!(
        "device {:?} dtype {dtype:?} batch {batch} n {n}: load {load:.1}s, preprocess {:.1} img/s, model {:.2} img/s, end-to-end {:.2} img/s",
        if device.is_metal() {
            "metal"
        } else if device.is_cuda() {
            "cuda"
        } else {
            "cpu"
        },
        n as f64 / pre_s,
        n as f64 / model_s,
        n as f64 / (pre_s + model_s),
    );
    Ok(())
}
