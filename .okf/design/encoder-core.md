---
type: Design
title: Encoder core — shared blocks and the Encoder trait for embedding families
description: The encoder crate's core layer every embedding model runs on — the Encoder trait (preprocess, forward, pool, L2), the core blocks with the families that call them, model files from the local Hugging Face cache, the exactness harness, cargo features, and what a family writes itself.
resource: src
tags: [embedding, encoder, candle, metal, cuda, architecture]
status: draft
generated:
  by: claude-code/claude-opus-5-5
  at: '2026-11-12T22:00:00Z'
sources:
  - id: tensorfold
    resource: https://github.com/ashhart/TensorFold/blob/main/docs/recipes/adding-a-zig-family.md
    title: TensorFold, "Adding a Zig family" — core blocks built once, a family writes config, weight map and layer graph, PRs justify skipped blocks
  - id: code
    resource: src
    title: The crate source (lib.rs, weights.rs, attention.rs, exact.rs, families/) and Cargo.toml
---

# Overview

The `candle-embeddings` crate holds image and text embedding models in candle 0.11.1. It follows TensorFold's
family pattern:[^tensorfold] a shared block is built once in the core; a model **family** writes only
its config, weight map and layer graph, plus kernels for layers only it has; every family is exact
against its reference on day one; and the [recipe](/recipes/adding-an-encoder-family.md) lists the
steps in order with their measured gains.[^code]

The first family is [EmbeddingGemma 2](/design/embeddinggemma2.md).

# The Encoder trait

```rust
pub trait Encoder {
    type Image;                                   // preprocessed on the CPU (Patches for ViTs)
    fn dim(&self) -> usize;
    fn device(&self) -> &Device;
    fn dtype(&self) -> DType;                     // compute dtype
    fn batch_size(&self) -> usize;
    fn pooling(&self) -> Pooling;                 // Mean
    fn preprocess(&self, img: &DynamicImage) -> Result<Self::Image>;
    fn forward_images(&self, batch: &[Self::Image], prompt: Option<&str>) -> Result<Hidden>;
    fn forward_texts(&self, batch: &[&str], prompt: Option<&str>) -> Result<Hidden>;
    // provided: finish (pool + L2 + non-finite check), embed_preprocessed, embed_images,
    // embed_image_paths, embed_texts — all split the input into batch_size chunks.
}
```

`Hidden` is the last hidden states (b, s, dim) in f32 with each row's valid length. The core pools
and normalizes, so a family never writes its own pooling or L2. A non-finite value fails the batch
(crashing beats corrupting). `Image` is a type the caller can build on a decode thread while the
device runs the previous batch.

# Core blocks

"Families" names the ones that call the block now.

| Block | The call a family makes | Families |
|---|---|---|
| `weights.rs` | `Checkpoint::new(path)`, `.var_builder(dtype, dev)` (mmap, tensors read on demand), `.token_table(name, vocab, hidden)` (rows read from the map, never the whole table), `linear_no_bias`, `model_dir(model, revision, files, download)`, `hub_cache_dir()` | embeddinggemma2 |
| `norm.rs` | `RmsNorm::new` (weight as stored), `RmsNorm::without_scale`; fused kernel, f32 accumulation | embeddinggemma2 |
| `rope.rs` | `tables_1d` + `apply` (text); `axial_inv_freq`, `tables_axial_2d` + `apply_axial_2d` (2-D ViT RoPE on the fused 1-D kernel) | embeddinggemma2 (text: 1-D, vision: axial) |
| `attention.rs` | `heads_first` (stride fix), `attention(q, k, v, mask)` (plain, GQA, f32 softmax), `attention_fused(q, k, v)` (per-backend: Metal `sdpa`; CUDA bf16/f16 FlashAttention-2 with the `flash-attn` feature; else plain), `padding_mask(lengths, s, window, dev)` | embeddinggemma2 (vision: fused; text: plain with mask) |
| `mlp.rs` | `Activation::from_config`, `GatedMlp { gate, up, down, act }` | embeddinggemma2 |
| `preprocess.rs` | `open_image` (EXIF), `resize_bicubic_aa` (torchvision uint8 bit-exact), `patchify(img, patch)`, `Patches`, `pixel_maxdiff` | embeddinggemma2 |
| `pool.rs` | `mean(states, lengths)`, `l2_normalize` (both through `Encoder::finish`), `grid_sum(h, w, h, k)` (ViT k x k pooling; the caller folds 1/k² into its next scale) | embeddinggemma2 |
| `exact.rs` | implement `ExactCheck` (`patch_size`, optional `image_input_ids`), then `exact::run(&enc, &Spec, verbose)` and `Report::check_f32()` for an exact path (cosine >= 0.99999 and max element difference <= 1e-4, so a bf16 result at 0.999784 fails) or `Report::check(min_cos)` for a half-precision path. Both fail on a NaN cosine and on empty fixtures | embeddinggemma2 |

**What a family writes.** Its config struct (refusing settings it does not implement), its weight map
(EmbeddingGemma 2: `Gemma4ClippableLinear` keeps weights at `<name>.linear.weight`), its layer
graphs, its input sequence (special tokens, soft-token placement), the parts of its image processor
that are its own (EmbeddingGemma 2: the aspect-ratio target size), and kernels for layers only it has
(none yet).

**What a family must justify.** For each core block it does not use, the family's module doc and the
change that adds it name the block and say why its own code is needed — or refuse that feature by
name until the block covers it. EmbeddingGemma 2 skips none.

# Model files

`weights::model_dir(model, revision, files, download)` accepts a local directory that holds the files,
or a Hugging Face id (`owner/name`). By default it reads the local Hugging Face cache directly, in the
layout `huggingface_hub` writes: the hub cache is `HF_HUB_CACHE`, else `HUGGINGFACE_HUB_CACHE`, else
`$HF_HOME/hub`, with `HF_HOME` defaulting to `$XDG_CACHE_HOME/huggingface`, then
`~/.cache/huggingface`; `models--{owner}--{name}/refs/{revision}` holds a commit hash and the files
are in `snapshots/{hash}/`. A miss is an error that names the fix:
`hf download <model> --revision <rev>`. Only with the `download` feature (hf-hub, off by default) does
a miss download the files. See [model cache decision](/decisions/model-cache.md).

# Rules

- Stage 0 comes first: the family passes `exact::run` against its transformers reference in f32
  before any speed step (`Report::check_f32`: something compared, every cosine finite, ids, pixels
  0/255, min cosine, retrieval equal to the reference). See the [recipe](/recipes/adding-an-encoder-family.md).
- A speed step keeps the f32 path exact. Lossy paths (bf16, fused kernels that change the summation
  order) are opt-in by dtype and report their cosine against the f32 reference.
- Kernels stay per backend behind one core call (`attention_fused` picks the kernel; a family never
  branches on the device).
- Source files stay under 600 lines.

# Cargo features

| Feature | Default | What |
|---|---|---|
| (Metal) | by target | candle's `metal` feature on every macOS build; no feature flag |
| `cuda` | off | candle's CUDA device, RMSNorm and RoPE kernels |
| `flash-attn` | off | implies `cuda`; adds `candle-flash-attn` 0.11.1 (FlashAttention-2 for bf16/f16), built with nvcc and `CUDA_COMPUTE_CAP`, needs network for CUTLASS |
| `download` | off | `hf-hub` 1.0 (blocking) to fetch a model missing from the local cache; adds a TLS/HTTP stack |

# Layout

```text
src/lib.rs                     Encoder, Hidden, Pooling
src/{weights,norm,rope,attention,mlp,preprocess,pool,exact}.rs   core blocks
src/families/embeddinggemma2/  config, preprocess (target size), vision, text, mod (loader, sequence)
tests/embeddinggemma2.rs       smoke (cargo test; prints SKIPPED to stderr when the model or references
                               are missing) + full_parity (ignored, the stage-0 gate); Metal on macOS,
                               an error when Metal is missing (EMEANS_ENCODER_DEVICE=cpu asks for the CPU)
examples/parity.rs, throughput.rs
bench/gemma/reference.py       regenerates the parity fixtures bench/gemma/images and bench/gemma/ref (gitignored)
```

Commands: `mise run check` (fmt, clippy with and without `download`, tests), `mise run test`,
`mise run exact` (the ignored gate: `cargo test --release --test embeddinggemma2 -- --ignored`).

[^tensorfold]: TensorFold `docs/recipes/adding-a-zig-family.md`, "Build each step once, in the core" and "What a family PR must justify".
[^code]: `src/` and `Cargo.toml`, 2026-11-12.
