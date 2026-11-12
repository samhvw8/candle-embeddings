# encoder

Image and text embedding encoders in candle: the `Encoder` trait and shared blocks in the core,
one module per model family under `src/families/`. First family: EmbeddingGemma 2. A public repo:
no model weights, no personal data.

## Layout

| Path | What |
|---|---|
| `src/lib.rs` | `Encoder` trait (`embed_images`, `embed_texts`, `embed_image_paths`), `Hidden`, `Pooling` |
| `src/{weights,norm,rope,attention,mlp,preprocess,pool,exact}.rs` | Shared blocks; `weights::model_dir` resolves models from the local HF cache |
| `src/families/embeddinggemma2/` | Config, preprocessing, vision and text towers, `EmbeddingGemma2` |
| `tests/embeddinggemma2.rs` | Smoke test (skips without model or fixtures) and the ignored exactness gate |
| `examples/` | `parity`, `throughput` |
| `bench/gemma/reference.py` | Writes the parity fixtures (`images/`, `ref/`, gitignored) from the official sentence-transformers path |
| `.okf/` | Design notes and the recipe for a new family (OKF bundle) |

## Commands

| Task | Command |
|---|---|
| Format, lint (default and `download`), test | `mise run check` |
| Exactness gate (needs the cached model and the fixtures) | `mise run exact` |
| Fixtures | `uv run --script bench/gemma/reference.py` |

## Rules

- Exact first: a family matches its reference (`exact.rs` harness) before any speed step; follow
  `.okf/recipes/adding-an-encoder-family.md`. f32 is the exact path; bf16 is opt-in.
- No network by default. `hf-hub` stays behind the `download` feature; the default build reads
  the local Hugging Face cache and fails with a clear message on a miss.
- No model weights or generated fixtures in git.
- Metal is chosen by target (macOS); `cuda` and `flash-attn` are features.
- Long GPU runs on a shared workstation: low priority, short chunks.

Project knowledge lives in `.okf/`. Update it in the same change as the code it describes, and
validate it (`okf_validate.py .okf --strict`).
