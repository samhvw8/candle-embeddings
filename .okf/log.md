# Update Log

## 2026-11-12
* **Creation**: Created this bundle for the encoder library from the knowledge notes of the photo tool it was split out of, scrubbed of personal data: results only on public test images, no private paths or host names. Paths updated to the standalone crate, and the new default of reading model files from the local Hugging Face cache (hf-hub behind the `download` feature) recorded — see [getting started](getting-started.md), [encoder core](design/encoder-core.md), [model cache](decisions/model-cache.md).
