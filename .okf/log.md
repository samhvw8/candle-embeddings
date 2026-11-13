# Update Log

## 2026-11-13
* **Vision token budget**: `EmbeddingGemma2::with_max_soft_tokens` (70, 140, 280, 560, 1120; default stays 280), `reference.py --max-soft-tokens N`, and a budget argument on the parity and throughput examples. Exact at 70 and 140; 4.3x / 2.1x faster on Metal f32 — see [EmbeddingGemma 2](design/embeddinggemma2.md#vision-token-budget).
* **Rename**: The crate and repository are now `candle-embeddings` (library name `candle_embeddings`, github.com/samhvw8/candle-embeddings). `encoder` was taken on crates.io and too generic to find — see [getting started](getting-started.md).

## 2026-11-12
* **Creation**: Created this bundle for the encoder library from the knowledge notes of the photo tool it was split out of, scrubbed of personal data: results only on public test images, no private paths or host names. Paths updated to the standalone crate, and the new default of reading model files from the local Hugging Face cache (hf-hub behind the `download` feature) recorded — see [getting started](getting-started.md), [encoder core](design/encoder-core.md), [model cache](decisions/model-cache.md).
