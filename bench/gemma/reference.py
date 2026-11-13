# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#   "torch==2.11.*",
#   "torchvision==0.26.*",
#   "transformers==5.22.2",
#   "sentence-transformers==6.3.0",
#   "pillow==12.*",
#   "numpy==2.*",
#   "scikit-image==0.26.*",
# ]
# ///
"""Reference EmbeddingGemma 2 embeddings for the Rust parity test.

Runs the official sentence-transformers path on CPU in float32 and writes:

  images/<name>.png        test images (scikit-image sample data, CC0 / public domain, plus crops)
  ref/images.json          per image: file, input_ids, pixel grid, soft-token count
  ref/images.f32           n x 768 image embeddings (no prompt, as the model card says)
  ref/pixels/<name>.f32    the processor's unpadded patches (num_patches x 768), for model-only parity
  ref/texts.json           texts, prompts and input_ids
  ref/texts_<prompt>.f32   text embeddings per prompt

Usage: uv run --script bench/gemma/reference.py   (run from the repo root)

With `--max-soft-tokens N` (70, 140, 560 or 1120; the model's default is 280) the image processor
runs at that vision token budget, and the outputs go to `soft<N>/ref/` with image paths relative to
`soft<N>/`, so the same exactness harness reads them (fixtures dir `bench/gemma/soft<N>`). Text
vectors do not depend on the budget; they are copied from `ref/` when it exists.
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
IMG_DIR = HERE / "images"
REF_DIR = HERE / "ref"
MODEL = "google/embeddinggemma-2"

# name -> (skimage loader, caption). Licenses are in skimage/data/_fetchers.py docstrings.
IMAGES = {
    "astronaut": ("astronaut", "an astronaut in a white spacesuit holding a helmet, with an american flag"),
    "cat": ("chelsea", "a ginger tabby cat lying down"),
    "coffee": ("coffee", "a cup of coffee on a saucer with a spoon"),
    "rocket": ("rocket", "a rocket launching into the sky"),
    "hubble": ("hubble_deep_field", "a deep field of distant galaxies in space"),
    "coins": ("coins", "a collection of old coins on a dark background"),
    "horse": ("horse", "a black silhouette of a horse on a white background"),
    "camera": ("camera", "a black and white photo of a photographer with a camera on a tripod"),
    "page": ("page", "a scanned page of printed text"),
    "grass": ("grass", "a close up texture of green grass"),
    "logo": ("logo", "the scikit-image logo, a colorful graphic with the word image"),
}
# Extra texts: one short, one long enough (> 512 tokens) to exercise the sliding-window mask.
EXTRA_TEXTS = [
    "What causes the northern lights?",
    " ".join(
        f"Paragraph {i}: the quick brown fox jumps over the lazy dog while the photographer "
        f"adjusts the tripod and the astronaut checks the oxygen supply before the launch."
        for i in range(40)
    ),
]
PROMPTS = {"clustering": "task: clustering | query: ", "query": "task: search result | query: "}


def make_images() -> None:
    import skimage.data as skd

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    for name, (loader, _) in IMAGES.items():
        path = IMG_DIR / f"{name}.png"
        if path.exists():
            continue
        arr = getattr(skd, loader)()
        Image.fromarray(arr).save(path)
    # Aspect-ratio edge cases derived from the samples above.
    coffee = Image.open(IMG_DIR / "coffee.png").convert("RGB")
    w, h = coffee.size
    variants = {
        "coffee_wide": (coffee.crop((0, h // 3, w, h // 3 + w // 4)), IMAGES["coffee"][1]),  # 4:1
        "page_strip": (Image.open(IMG_DIR / "page.png").crop((0, 0, 380, 30)), IMAGES["page"][1]),  # tiny, 0-row branch
        "rocket_tall": (Image.open(IMG_DIR / "rocket.png").resize((90, 1200)), IMAGES["rocket"][1]),  # 1:13
        "cat_small": (Image.open(IMG_DIR / "cat.png").resize((64, 48)), IMAGES["cat"][1]),  # upscaled
    }
    for name, (img, caption) in variants.items():
        path = IMG_DIR / f"{name}.png"
        if not path.exists():
            img.save(path)
        IMAGES[name] = (None, caption)


DEFAULT_SOFT_TOKENS = 280


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-soft-tokens", type=int, default=DEFAULT_SOFT_TOKENS)
    budget = ap.parse_args().max_soft_tokens
    torch.set_num_threads(int(os.environ.get("REF_THREADS", "4")))
    make_images()
    if budget == DEFAULT_SOFT_TOKENS:
        fixtures, ref_dir = HERE, REF_DIR
    else:
        fixtures = HERE / f"soft{budget}"
        ref_dir = fixtures / "ref"
    ref_dir.mkdir(parents=True, exist_ok=True)
    (ref_dir / "pixels").mkdir(exist_ok=True)

    from sentence_transformers import SentenceTransformer

    t0 = time.time()
    model = SentenceTransformer(
        MODEL,
        device="cpu",
        model_kwargs={"dtype": torch.float32},
        config_kwargs={"audio_config": None},
    )
    model.eval()
    print(f"loaded in {time.time() - t0:.1f}s; dtype {next(model.parameters()).dtype}", flush=True)
    processor = model[0].processor
    print("image processor:", type(processor.image_processor).__name__, flush=True)
    # The vision token budget: Gemma4ImageProcessor.max_soft_tokens, one of (70, 140, 280, 560, 1120).
    processor.image_processor.max_soft_tokens = budget
    print("max_soft_tokens:", processor.image_processor.max_soft_tokens, flush=True)

    names = list(IMAGES)
    img_meta = []
    img_embs = []
    for name in names:
        img = Image.open(IMG_DIR / f"{name}.png")
        # What the Transformer module feeds the model for one image.
        feats = model.preprocess([img])
        ids = feats["input_ids"][0].tolist()
        pos = feats["image_position_ids"][0]
        valid = (pos != -1).all(dim=-1)
        patches = feats["pixel_values"][0][valid].to(torch.float32).numpy()
        patches.astype("<f4").tofile(ref_dir / "pixels" / f"{name}.f32")
        grid_w = int(pos[valid][:, 0].max()) + 1
        grid_h = int(pos[valid][:, 1].max()) + 1
        # The budget took effect: padded to budget * 9 patches, at most budget soft tokens.
        assert pos.shape[0] == budget * 9, (name, pos.shape)
        assert len(ids) - 4 <= budget, (name, len(ids))
        with torch.no_grad():
            emb = model.encode([img], batch_size=1, convert_to_numpy=True, normalize_embeddings=False)[0]
        img_embs.append(emb.astype(np.float32))
        w, h = Image.open(IMG_DIR / f"{name}.png").size
        img_meta.append(
            {
                "name": name,
                "file": os.path.relpath(IMG_DIR / f"{name}.png", fixtures),
                "size": [w, h],
                "grid": [grid_w, grid_h],
                "num_patches": int(valid.sum()),
                "input_ids": ids,
                "caption": IMAGES[name][1],
                "norm": float(np.linalg.norm(emb)),
            }
        )
        print(f"{name}: {w}x{h} -> grid {grid_w}x{grid_h}, {len(ids)} tokens", flush=True)
    np.stack(img_embs).astype("<f4").tofile(ref_dir / "images.f32")
    (ref_dir / "images.json").write_text(json.dumps(img_meta, indent=1))

    captions = sorted({IMAGES[n][1] for n in IMAGES})
    texts = captions + EXTRA_TEXTS
    base_texts = REF_DIR / "texts.json"
    if ref_dir != REF_DIR and base_texts.is_file() and json.loads(base_texts.read_text())["texts"] == texts:
        for key in PROMPTS:
            shutil.copy(REF_DIR / f"texts_{key}.f32", ref_dir / f"texts_{key}.f32")
        shutil.copy(base_texts, ref_dir / "texts.json")
        print("texts: copied from ref/ (they do not depend on the image budget)", flush=True)
    else:
        text_meta = {"texts": texts, "prompts": PROMPTS, "input_ids": {}}
        for key, prompt in PROMPTS.items():
            with torch.no_grad():
                embs = model.encode(texts, prompt=prompt, batch_size=2, convert_to_numpy=True)
            embs.astype("<f4").tofile(ref_dir / f"texts_{key}.f32")
            feats = model.preprocess([prompt + t for t in texts[:1]])
            text_meta["input_ids"][key] = feats["input_ids"][0].tolist()
        (ref_dir / "texts.json").write_text(json.dumps(text_meta, indent=1))

    # Retrieval sanity check: each image's nearest caption.
    ie = np.stack(img_embs)
    ie /= np.linalg.norm(ie, axis=1, keepdims=True)
    te = np.fromfile(ref_dir / "texts_query.f32", dtype="<f4").reshape(len(texts), -1)
    sims = ie @ te.T
    hits = 0
    for i, m in enumerate(img_meta):
        top = texts[int(sims[i].argmax())]
        hits += top == m["caption"]
        print(f"  {m['name']:12s} -> {top[:60]}", flush=True)
    print(f"retrieval (query prompt, {budget} soft tokens): {hits}/{len(img_meta)} top-1", flush=True)
    print(f"done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
