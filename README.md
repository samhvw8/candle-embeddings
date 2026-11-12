# encoder

Image and text embedding encoders in Rust, on [candle](https://github.com/huggingface/candle):
a small `Encoder` trait, shared transformer blocks, and one module per model family. The first
family is **EmbeddingGemma 2** (`google/embeddinggemma-2`), 768-d embeddings for images and text
in one space, checked against the official sentence-transformers outputs.

## Usage

```toml
[dependencies]
encoder = { git = "https://github.com/samhvw8/encoder" }                         # CPU, plus Metal on macOS
# encoder = { git = "...", features = ["cuda"] }                                  # plus CUDA
# encoder = { git = "...", features = ["download"] }                              # fetch missing models
```

```rust
use candle_core::Device;
use encoder::Encoder;
use encoder::families::embeddinggemma2::{EmbeddingGemma2, MODEL_ID};

let device = Device::new_metal(0)?;                  // or Device::Cpu, Device::new_cuda(0)?
let model = EmbeddingGemma2::load(MODEL_ID, &device)?;
let images = model.embed_image_paths(&["cat.jpg", "dog.png"], None)?; // Vec<Vec<f32>>, unit rows
let texts = model.embed_texts(&["a cat on a sofa"], None)?;
```

`load` takes a Hugging Face id or a local directory with `config.json`, `tokenizer.json` and
`model.safetensors`.

## Models are read from the local Hugging Face cache

By default the crate does not touch the network. It reads the cache that `huggingface_hub` and
the `hf` CLI write: `HF_HUB_CACHE`, else `$HF_HOME/hub` (default `~/.cache/huggingface/hub`),
`models--{owner}--{name}/refs/{revision}` → `snapshots/{hash}/`. `MODEL_ID` is pinned to the
revision the parity numbers were measured on. When the model is missing, the error names the
command that fetches it:

```sh
hf download google/embeddinggemma-2 --revision <MODEL_REVISION>
```

With the `download` feature, a cache miss downloads through `hf-hub` instead. It is off by
default: `hf-hub` adds an HTTP/TLS stack (tokio, reqwest, rustls, aws-lc) of about 175 crates:
220 unique dependencies without it, 395 with it (`cargo tree -e normal,build` on macOS).

## Features

| Feature | Effect |
|---|---|
| (macOS) | candle's Metal backend is always on; there is no `metal` feature |
| `cuda` | candle's CUDA backend |
| `flash-attn` | FlashAttention-2 for bf16/f16 attention on CUDA (sm_80+; builds with `nvcc`, needs `CUDA_COMPUTE_CAP`, a long first build) |
| `download` | fetch models that are not in the local cache (`hf-hub`) |

## Exactness

A family is exact against its reference before it is fast. `tests/embeddinggemma2.rs` compares
image and text embeddings with the outputs of the official sentence-transformers path
(`bench/gemma/reference.py`, CPU f32). Create the fixtures, then run the gate:

```sh
uv run --script bench/gemma/reference.py                              # writes bench/gemma/{images,ref}
cargo test --release --test embeddinggemma2 -- --ignored              # needs the cached model
```

`cargo test` runs a smoke test that skips when the model or the fixtures are missing.

## Adding a family

Write the config, the weight map and the layer graph under `src/families/<name>/`, reuse the core
blocks, and implement `Encoder`. A family names each core block it does not use, and why. The
step-by-step recipe (exact first, then the speed steps in order, with measured gains) is in
[`.okf/recipes/`](.okf/index.md).

## Limits

- One family so far.
- Metal is tested on an M1 Pro, CUDA on one RTX 3060 (Linux). bf16 on Metal and CUDA is faster and
  not bit-exact; f32 is the exact path.
- No model weights are included. EmbeddingGemma 2 weights are subject to Google's Gemma terms.
- Version 0.1: the API may change. No crates.io release yet.

## Development

```sh
mise run check   # fmt, clippy -D warnings (default and download), tests
mise run exact   # the exactness gate (needs the cached model and bench/gemma/ref)
```

## License

MIT OR Apache-2.0, at your option. See `NOTICE` for the model and test-image terms.
