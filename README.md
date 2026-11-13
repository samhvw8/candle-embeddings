# encoder

Image and text embeddings in Rust, without Python. encoder runs **EmbeddingGemma 2**
(`google/embeddinggemma-2`) on [candle](https://github.com/huggingface/candle): images and texts
become 768-d unit vectors in one shared space. It runs on Apple GPUs (Metal), NVIDIA GPUs (CUDA) or
the CPU.

- **Same vectors as the official pipeline.** In f32, on 15 test images and 26 text inputs, the
  minimum cosine against sentence-transformers is 1.000000 on Metal and on the CPU
  ([parity](.okf/design/embeddinggemma2.md#parity)).
- **Faster than sentence-transformers on the same GPU**, in every setting measured. RTX 3060, bf16
  with FlashAttention-2: **10.31 images/s** at batch 8, against 9.25
  ([recipe, step 5](.okf/recipes/adding-an-encoder-family.md#stage-1-speed-in-order)).
- **No network by default.** It reads models from the local Hugging Face cache. Downloading is an
  opt-in feature.

> Version 0.1. Not on crates.io yet: depend on the git repository. The API may change.
> No model weights are included; see [model terms](#license-and-credit).

## Contents

[Quickstart](#quickstart) ·
[Choosing a dtype](#choosing-a-dtype) ·
[Benchmarks](#benchmarks) ·
[Cargo features](#cargo-features) ·
[Limits](#limits) ·
[Docs](#docs) ·
[License](#license-and-credit)

## Quickstart

**1. Get the model once.** The default build never downloads. Fetch the pinned revision with the
`hf` CLI:

```sh
hf download google/embeddinggemma-2 --revision 914f7f89142e33e77833254d9c9b90c3cef7303b
```

encoder reads `HF_HUB_CACHE`, else `$HF_HOME/hub` (default `~/.cache/huggingface/hub`). When the
model is missing, the error names this command. To download on a cache miss instead, turn on the
`download` feature.

**2. Add the crate.**

```toml
[dependencies]
encoder = { git = "https://github.com/samhvw8/encoder" }                         # CPU, plus Metal on macOS
# encoder = { git = "https://github.com/samhvw8/encoder", features = ["cuda"] }  # plus CUDA
candle-core = "0.11.1"                                                            # for Device
```

**3. Embed.**

```rust
use candle_core::Device;
use encoder::Encoder;
use encoder::families::embeddinggemma2::{EmbeddingGemma2, MODEL_ID, prompts};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_metal(0)?; // or Device::Cpu, or Device::new_cuda(0)? (feature cuda)
    let model = EmbeddingGemma2::load(MODEL_ID, &device)?; // f32: the exact path

    // Images: no prompt. Texts: a task prompt from `prompts`.
    let images = model.embed_image_paths(&["cat.jpg", "dog.png"], None)?; // 768-d unit vectors
    let query = model.embed_texts(&["a cat on a sofa"], Some(prompts::SEARCH_QUERY))?;

    // Unit vectors: the dot product is the cosine similarity.
    let score: f32 = images[0].iter().zip(&query[0]).map(|(a, b)| a * b).sum();
    println!("cat.jpg vs query: {score:.3}");
    Ok(())
}
```

**Prompts.** The model card says to pass images without a prefix. Texts take a task prefix:
`prompts::SEARCH_QUERY`, `DOCUMENT`, `CLUSTERING`, `CLASSIFICATION` or `SENTENCE_SIMILARITY`.

Other entry points:

| Call | Use it to |
|---|---|
| `EmbeddingGemma2::load(model, &device)` | Load from a Hugging Face id or a local directory with `config.json`, `tokenizer.json` and `model.safetensors` |
| `load_with(model, &device, dtype)` | Pick the dtype (see below) |
| `with_batch_size(n)` | Set inputs per forward pass (default 1) |
| `embed_images(&[DynamicImage], prompt)` | Embed decoded images |
| `preprocess(&img)`, then `embed_preprocessed(&[Patches], prompt)` | Preprocess on the CPU on another thread while the GPU runs the previous batch |
| `encoder::open_image(path)` | Open an image and apply its EXIF orientation |

## Choosing a dtype

f32 is the default and the exact path. Cosine against the f32 CPU reference, on 15 images and 26
texts ([dtype](.okf/design/embeddinggemma2.md#dtype)):

| dtype | Image cosine, min | M1 Pro, images/s (batch 4) | M1 Pro, peak memory | Use it when |
|---|---|---|---|---|
| f32 | 1.000000 | 1.55 | 3.07 GB | You need the reference vectors (default) |
| bf16 | 0.999784 | 1.59 | 1.39 GB | Memory is tight on a Mac, or you want speed on CUDA (with `flash-attn`) |
| f16 | 0.999950 | 1.81 | 2.02 GB | Not recommended: the model card warns that activations can exceed the f16 range. A non-finite output fails the batch |

On an RTX 3060, bf16 with FlashAttention-2 gave a minimum image cosine of 0.999816
([recipe, step 5](.okf/recipes/adding-an-encoder-family.md#stage-1-speed-in-order)).

## Benchmarks

End to end: decode, preprocess and model, on public test images. Images per second at batch 1 /
batch 8.

| Hardware, dtype | encoder | sentence-transformers | Source |
|---|---|---|---|
| RTX 3060, bf16 | **9.67 / 10.31** (with `flash-attn`) | 5.41 / 9.25 | [recipe, step 5](.okf/recipes/adding-an-encoder-family.md#stage-1-speed-in-order) |
| RTX 3060, f32 | **3.24 / 3.28** | 2.85 / 3.17 | [throughput](.okf/design/embeddinggemma2.md#throughput) |
| M1 Pro, f32 | **1.43 / 1.45** (Metal) | 0.98 / out of memory (MPS); 0.16 / 0.21 (CPU) | [throughput](.okf/design/embeddinggemma2.md#throughput) |

Other processes shared the M1 Pro GPU during the runs, so treat its rates as a floor. The vision
tower takes about 90% of the time.

## Cargo features

| Feature | Effect |
|---|---|
| (none on macOS) | candle's Metal backend is always on; there is no `metal` feature |
| `cuda` | candle's CUDA backend |
| `flash-attn` | FlashAttention-2 for bf16 and f16 attention on CUDA (sm_80+, implies `cuda`). Needs `nvcc`, `CUDA_COMPUTE_CAP` and network access for CUTLASS. The first build took 18 minutes on a 2x Xeon box |
| `download` | Fetch a model that is not in the local cache (`hf-hub`). Off by default: it adds an HTTP and TLS stack, 395 dependencies instead of 220 ([decision](.okf/decisions/model-cache.md)) |

## Limits

- **One model family.** EmbeddingGemma 2 only. New families follow the
  [recipe](.okf/recipes/adding-an-encoder-family.md).
- **Not on crates.io.** Depend on the git repository.
- **No model weights included.** EmbeddingGemma 2 weights are subject to Google's Gemma terms.
- **Tested hardware:** Metal on an M1 Pro, CUDA on one RTX 3060 (Linux). Other GPUs and Windows
  are untested.
- **bf16 and f16 are not exact.** f32 is the only path that matches the reference.
- **FlashAttention covers the vision tower only.** The text encoder (masks, head_dim 512) runs the
  plain attention path.
- Rust only: no language bindings.

## Docs

| To | Read |
|---|---|
| Understand the crate and its modules | [`.okf/getting-started.md`](.okf/getting-started.md) |
| See the `Encoder` trait and the shared blocks | [`.okf/design/encoder-core.md`](.okf/design/encoder-core.md) |
| See how EmbeddingGemma 2 works: inputs, preprocessing, parity, throughput | [`.okf/design/embeddinggemma2.md`](.okf/design/embeddinggemma2.md) |
| Add a model family: exact first, then the speed steps with measured gains | [`.okf/recipes/adding-an-encoder-family.md`](.okf/recipes/adding-an-encoder-family.md) |
| Know why downloading is opt-in | [`.okf/decisions/model-cache.md`](.okf/decisions/model-cache.md) |

To cluster the vectors, see [emeans](https://github.com/samhvw8/emeans).

## Development

```sh
mise run check                                  # fmt, clippy -D warnings (default and download), tests
uv run --script bench/gemma/reference.py        # parity fixtures from sentence-transformers (CPU, f32)
mise run exact                                  # the exactness gate (needs the cached model and the fixtures)
cargo run --release --example parity -- metal f32 4
```

`cargo test` runs a smoke test that skips when the model or the fixtures are missing.

## License and credit

MIT OR Apache-2.0, at your option ([LICENSE-MIT](LICENSE-MIT), [LICENSE-APACHE](LICENSE-APACHE)).

- The model architecture follows the reference implementation in Hugging Face transformers
  (Apache-2.0). The vision tower keeps the structure of candle's `gemma4` module.
- EmbeddingGemma 2 weights are not included. They are subject to Google's
  [Gemma terms of use](https://ai.google.dev/gemma/terms).
- The parity test images are scikit-image sample data (public domain or CC0). The reference script
  generates them; the repository does not store them.

See [NOTICE](NOTICE).
