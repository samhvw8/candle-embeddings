---
type: Decision
title: Model files from the local Hugging Face cache; downloading is opt-in
description: The encoder crate reads model files straight from the local Hugging Face cache by default and only downloads with the off-by-default `download` feature (hf-hub), which cuts the default dependency tree from 395 to 220 packages.
resource: src/weights.rs
tags: [decision, dependencies, hugging-face, weights, build-time]
status: draft
generated:
  by: claude-code/claude-opus-5-5
  at: '2026-11-12T22:00:00Z'
sources:
  - id: code
    resource: src/weights.rs
    title: model_dir, hub_cache_dir, cached_snapshot, fetch (feature download)
  - id: hf-cache
    resource: https://huggingface.co/docs/huggingface_hub/guides/manage-cache
    title: huggingface_hub cache layout (models--owner--name, refs, snapshots) and HF_HOME / HF_HUB_CACHE
---

# Decision

`weights::model_dir` resolves a model without network access by default: a local directory, or the
Hugging Face hub cache read directly in the layout `huggingface_hub` writes.[^hf-cache]

| Step | Rule |
|---|---|
| Hub cache | `HF_HUB_CACHE`, else `HUGGINGFACE_HUB_CACHE`, else `$HF_HOME/hub`; `HF_HOME` defaults to `$XDG_CACHE_HOME/huggingface`, else `~/.cache/huggingface` (empty variables are ignored) |
| Snapshot | `models--{owner}--{name}/refs/{revision}` holds the commit hash; the files are in `snapshots/{hash}/`; every requested file must exist |
| Miss, default build | An error that names the fix: `hf download <model> --revision <rev>`, or build with `download` |
| Miss, `download` feature | `hf-hub` 1.0 (blocking) fetches the pinned revision into the same cache |

EmbeddingGemma 2 pins its revision (`MODEL_REVISION`); `resolve_model_dir` downloads on a miss only
when the feature is on, `cached_model_dir` never downloads.

# Why

- **Dependencies.** Unique packages in the normal and build dependency tree
  (`cargo tree -e normal,build --prefix none`, deduplicated): 220 without `download`, 395 with it.
  hf-hub brings a TLS and HTTP stack that a library user who already has the model (or fetches it
  with the `hf` CLI) never needs.
- **No surprise network access.** A library that silently downloads 1.4 GB on first use is hard to
  run in CI and on locked-down machines; the error message gives the one command to run instead.
- **Same cache.** Reading the `huggingface_hub` layout means a model fetched by Python tools or the
  `hf` CLI is found as is.

# Checks

A unit test in `weights.rs` builds a fake cache under `HF_HUB_CACHE` and checks both the hit and the
miss message. `mise run check` runs clippy with and without `download`.

[^hf-cache]: huggingface_hub, "Manage the cache".
