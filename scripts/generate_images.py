"""
generate_images.py
=====================
Generate images for every prompt in prompts.jsonl using Stable Diffusion XL.

Defaults are tuned for one A100/A6000-class GPU. For T4 or 16GB cards, pass
--low-vram (enables CPU offload + slicing) and expect ~3-4× slower throughput.

Outputs
-------
outputs/images/{category}/{template_idx}_{sample_idx}.jpg
outputs/generation_log.jsonl   - one line per generated image, with metadata
outputs/generation_failed.jsonl - prompts that failed (resumable)

Resumability
------------
Already-generated files are skipped on subsequent runs. You can interrupt
with Ctrl-C and re-run; it picks up where it left off.

Usage
-----
    pip install torch diffusers transformers accelerate safetensors pillow tqdm
    python generate_images.py
    python generate_images.py --low-vram        # for <24GB GPUs
    python generate_images.py --batch-size 4    # tune for your VRAM
    python generate_images.py --limit 50        # quick smoke test
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path
from typing import Iterator

import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_STEPS = 30        # DPM++ 2M Karras converges well at 25-30
DEFAULT_GUIDANCE = 7.0    # SDXL responds well to 5-8; 7 is a safe default
DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512
DEFAULT_BATCH_SIZE = 4    # bump up if you have headroom; 4 fits on 24GB at 1024
JPEG_QUALITY = 92         # quality/space tradeoff; 90-95 is imperceptible

OUT = Path("outputs")
IMG_DIR = OUT / "images"
LOG_PATH = OUT / "generation_log.jsonl"
FAIL_PATH = OUT / "generation_failed.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_SAFE_NAME = re.compile(r"[^a-z0-9]+")


def slugify(s: str) -> str:
    """Filesystem-safe directory name from a category."""
    return _SAFE_NAME.sub("_", s.lower()).strip("_")


def output_path(category: str, template_idx: int, sample_idx: int) -> Path:
    return IMG_DIR / slugify(category) / f"t{template_idx}_s{sample_idx:03d}.jpg"


def load_prompts(path: Path, limit: int | None) -> list[dict]:
    prompts = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line))
            if limit and len(prompts) >= limit:
                break
    return prompts


def filter_done(prompts: list[dict]) -> list[dict]:
    """Remove prompts whose output file already exists."""
    todo = []
    for p in prompts:
        out = output_path(p["category"], p["template_idx"], p["sample_idx"])
        if not out.exists():
            todo.append(p)
    return todo


def batched(items: list[dict], n: int) -> Iterator[list[dict]]:
    for i in range(0, len(items), n):
        yield items[i:i + n]


# ---------------------------------------------------------------------------
# Pipeline setup
# ---------------------------------------------------------------------------
def load_pipeline(low_vram: bool, dtype: torch.dtype) -> StableDiffusionXLPipeline:
    print(f"Loading {MODEL_ID} (dtype={dtype}, low_vram={low_vram})...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
        use_safetensors=True,
        add_watermarker=False,
    )

    # DPM++ 2M Karras: high-quality output in 25-30 steps (vs 50 for default)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True,
    )

    if low_vram:
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
        print("  CPU offload + VAE slicing enabled.")
    else:
        pipe = pipe.to("cuda")
        # SDPA is the default attention backend in modern torch and is fast.
        # xformers is no longer needed and was removed from many setups.

    pipe.set_progress_bar_config(disable=True)  # tqdm handles outer progress
    return pipe


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate_batch(
    pipe: StableDiffusionXLPipeline,
    batch: list[dict],
    steps: int,
    guidance: float,
    height: int,
    width: int,
) -> list[Image.Image]:
    """Generate one batch with per-prompt seeds."""
    # Per-prompt generators reproduce each image deterministically from its seed.
    # This lets us regenerate any single image later without redoing the run.
    device = pipe._execution_device
    generators = [
        torch.Generator(device=device).manual_seed(p["seed"])
        for p in batch
    ]

    with torch.inference_mode():
        result = pipe(
            prompt=[p["prompt"] for p in batch],
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=height,
            width=width,
            generator=generators,
            output_type="pil",
        )

    return result.images


def save_image(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="JPEG", quality=JPEG_QUALITY, optimize=True)


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def run(
    prompts_path: Path,
    batch_size: int,
    steps: int,
    guidance: float,
    height: int,
    width: int,
    low_vram: bool,
    limit: int | None,
) -> None:
    OUT.mkdir(exist_ok=True)
    IMG_DIR.mkdir(exist_ok=True)

    all_prompts = load_prompts(prompts_path, limit)
    todo = filter_done(all_prompts)

    n_total = len(all_prompts)
    n_skip = n_total - len(todo)
    print(f"Total prompts:     {n_total}")
    print(f"Already generated: {n_skip}")
    print(f"To generate:       {len(todo)}")

    if not todo:
        print("Nothing to do.")
        return

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    if not torch.cuda.is_available():
        print("WARNING: no CUDA detected. Generation on CPU will be unusably slow.")

    pipe = load_pipeline(low_vram=low_vram, dtype=dtype)

    t_start = time.time()
    n_done = 0
    n_fail = 0

    with tqdm(total=len(todo), unit="img", desc="generating") as pbar:
        for batch in batched(todo, batch_size):
            try:
                images = generate_batch(pipe, batch, steps, guidance, height, width)
            except torch.cuda.OutOfMemoryError:
                # Recover and retry one at a time
                torch.cuda.empty_cache()
                gc.collect()
                images = []
                for p in batch:
                    try:
                        images.extend(generate_batch(
                            pipe, [p], steps, guidance, height, width
                        ))
                    except Exception as e:
                        append_jsonl(FAIL_PATH, {**p, "error": str(e)})
                        images.append(None)
                        n_fail += 1
            except Exception as e:
                # Unrecoverable batch failure — log all and move on
                for p in batch:
                    append_jsonl(FAIL_PATH, {**p, "error": str(e)})
                n_fail += len(batch)
                pbar.update(len(batch))
                continue

            for p, img in zip(batch, images):
                if img is None:
                    pbar.update(1)
                    continue
                out = output_path(p["category"], p["template_idx"], p["sample_idx"])
                save_image(img, out)
                append_jsonl(LOG_PATH, {
                    "category": p["category"],
                    "template_idx": p["template_idx"],
                    "sample_idx": p["sample_idx"],
                    "prompt": p["prompt"],
                    "seed": p["seed"],
                    "path": str(out.relative_to(OUT)),
                })
                n_done += 1
                pbar.update(1)

            # Show running throughput in the bar
            elapsed = time.time() - t_start
            rate = n_done / elapsed if elapsed > 0 else 0
            pbar.set_postfix(rate=f"{rate:.2f}img/s", failed=n_fail)

    elapsed = time.time() - t_start
    print()
    print(f"Done. Generated {n_done} images in {elapsed/60:.1f} min "
          f"({n_done/elapsed:.2f} img/s).")
    if n_fail:
        print(f"Failures: {n_fail}  (see {FAIL_PATH})")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=str(OUT / "prompts.jsonl"))
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE)
    ap.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    ap.add_argument("--low-vram", action="store_true",
                    help="Enable CPU offload + VAE slicing for <24GB GPUs.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N prompts (smoke test).")
    args = ap.parse_args()

    run(
        prompts_path=Path(args.prompts),
        batch_size=args.batch_size,
        steps=args.steps,
        guidance=args.guidance,
        height=args.height,
        width=args.width,
        low_vram=args.low_vram,
        limit=args.limit,
    )


if __name__ == "__main__":
    sys.exit(main())