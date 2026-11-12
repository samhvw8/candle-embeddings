---
type: Reference
title: Getting started — encoder
description: What the encoder library is, its module map, features, how to get the model and run the exactness gate, and where to read next.
tags: [getting-started, embedding, candle, embeddinggemma-2]
status: draft
generated:
  by: claude-code/claude-opus-5-5
  at: '2026-11-12T22:00:00Z'
---

# Overview

`encoder` is one Rust crate that turns images and texts into embedding vectors with candle 0.11.1,
on Metal (every macOS build), CUDA (feature `cuda`) or the CPU. A small `Encoder` trait and shared
transformer blocks live in the core; each model is a **family** under `src/families/`. The first
family is EmbeddingGemma 2 (`google/embeddinggemma-2`, 768-d unit vectors), exact against the
sentence-transformers reference in f32.

```rust
use encoder::Encoder;
use encoder::families::embeddinggemma2::{EmbeddingGemma2, MODEL_ID, prompts};
let emb = EmbeddingGemma2::load(MODEL_ID, &Device::new_metal(0)?)?;
let images = emb.embed_image_paths(&paths, None)?;
let texts = emb.embed_texts(&["a cat"], Some(prompts::SEARCH_QUERY))?;
```

# Module map

| Path | What | Read |
|---|---|---|
| `src/lib.rs` | `Encoder`, `Hidden`, `Pooling` | [encoder core](design/encoder-core.md) |
| `src/{weights,norm,rope,attention,mlp,preprocess,pool,exact}.rs` | Core blocks: weights and the HF cache, RMSNorm, RoPE, attention (plain and fused), gated MLP, bit-exact preprocessing, pooling and L2, the exactness harness | [encoder core](design/encoder-core.md) |
| `src/families/embeddinggemma2/` | Config, target size, vision and text towers, loader and input sequence | [EmbeddingGemma 2](design/embeddinggemma2.md) |
| `tests/embeddinggemma2.rs` | `smoke` and the ignored `full_parity` gate | [recipe](recipes/adding-an-encoder-family.md) |
| `examples/parity.rs`, `examples/throughput.rs` | Parity per input on any device/dtype/batch; end-to-end throughput | [EmbeddingGemma 2](design/embeddinggemma2.md) |
| `bench/gemma/reference.py` | Regenerates the parity fixtures `bench/gemma/images` and `bench/gemma/ref` (public scikit-image samples; gitignored) | [recipe](recipes/adding-an-encoder-family.md) |

# Features

`cuda`, `flash-attn` (implies `cuda`), `download` (hf-hub; off by default). Metal comes by target.
By default the model is read from the local Hugging Face cache; fetch it once with
`hf download google/embeddinggemma-2 --revision 914f7f89142e33e77833254d9c9b90c3cef7303b`
([model cache](decisions/model-cache.md)). No model weights are in the repository.

# Commands

| Task | Command |
|---|---|
| Format, lint (with and without `download`), test | `mise run check` |
| Tests (the parity test skips with a message when the model or fixtures are missing) | `mise run test` |
| Parity fixtures | `uv run --script bench/gemma/reference.py` |
| Exactness gate | `mise run exact` |

# Where to read next

- [Encoder core](design/encoder-core.md), then [EmbeddingGemma 2](design/embeddinggemma2.md).
- Adding a model: [adding an encoder family](recipes/adding-an-encoder-family.md).
- Why downloading is opt-in: [model cache](decisions/model-cache.md).
