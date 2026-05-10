"""
03_generate_images_sd3.py
=========================
Generate images for every prompt in prompts.jsonl using Stable Diffusion 3 Medium.

Tuned for a single 24GB GPU (e.g., RTX 4090, RTX 3090, A5000). SD 3 Medium
uses three text encoders concurrently — CLIP-ViT-L/14, OpenCLIP-bigG/14,
and T5-XXL — and the T5 is the memory hog. Defaults below keep the model
on GPU by loading T5 in 8-bit; this is the configuration HF officially
recommends for 24GB.

If you have <24GB, pass --low-vram for CPU offload (~3-4× slower).
If you have plenty of headroom, pass --no-quantize for full fp16 T5.

Outputs
-------
outputs/images_sd3/{category}/t{template_idx}_s{sample_idx}.jpg
outputs/generation_log_sd3.jsonl
outputs/generation_failed_sd3.jsonl

Resumability
------------
Already-generated files are skipped. Interrupt and re-run; picks up where left off.

Usage
-----
    pip install torch diffusers transformers accelerate safetensors pillow tqdm bitsandbytes
    huggingface-cli login   # SD 3 requires accepting the license on HF
    python 03_generate_images_sd3.py
    python 03_generate_images_sd3.py --batch-size 1     # default; bump if you can
    python 03_generate_images_sd3.py --limit 10         # smoke test
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
from diffusers import StableDiffusion3Pipeline
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"

# SD 3 Medium settings - notably different from SDXL:
DEFAULT_STEPS = 28        # SD 3 converges in 25-30 steps
DEFAULT_GUIDANCE = 4.5    # SD 3 prefers MUCH lower CFG than SDXL (4-5 vs 7)
DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512
DEFAULT_BATCH_SIZE = 4    # SD 3 + 24GB headroom is tight; start at 1
JPEG_QUALITY = 92

OUT = Path("outputs")
IMG_DIR = OUT / "images_sd3"
LOG_PATH = OUT / "generation_log_sd3.jsonl"
FAIL_PATH = OUT / "generation_failed_sd3.jsonl"


# ---------------------------------------------------------------------------
# Helpers (unchanged from SDXL script)
# ---------------------------------------------------------------------------
_SAFE_NAME = re.compile(r"[^a-z0-9]+")


def slugify(s: str) -> str:
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
    return [
        p for p in prompts
        if not output_path(p["category"], p["template_idx"], p["sample_idx"]).exists()
    ]


def batched(items: list[dict], n: int) -> Iterator[list[dict]]:
    for i in range(0, len(items), n):
        yield items[i:i + n]


# ---------------------------------------------------------------------------
# Pipeline setup — this is where SD 3 differs significantly from SDXL
# ---------------------------------------------------------------------------
def load_pipeline(
    low_vram: bool,
    quantize_t5: bool,
    dtype: torch.dtype,
) -> StableDiffusion3Pipeline:
    print(f"Loading {MODEL_ID}")
    print(f"  dtype={dtype}, low_vram={low_vram}, quantize_t5={quantize_t5}")

    if quantize_t5 and not low_vram:
        # 8-bit T5 keeps the encoder on GPU at ~5GB instead of ~10GB.
        # This is the standard 24GB recipe.
        from transformers import T5EncoderModel, BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        text_encoder_3 = T5EncoderModel.from_pretrained(
            MODEL_ID,
            subfolder="text_encoder_3",
            quantization_config=quant_config,
            torch_dtype=dtype,
        )
        pipe = StableDiffusion3Pipeline.from_pretrained(
            MODEL_ID,
            text_encoder_3=text_encoder_3,
            torch_dtype=dtype,
        )
        # 8-bit T5 cannot be moved with .to("cuda"); it's already on GPU.
        # Move only the rest of the pipeline.
        pipe.to("cuda")
        print("  T5 loaded in 8-bit; rest of pipeline on CUDA.")

    elif low_vram:
        # CPU offload: each component swaps to GPU only when in use.
        # Slowest but works on small VRAM.
        pipe = StableDiffusion3Pipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
        )
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
        print("  CPU offload + VAE slicing enabled.")

    else:
        # Full fp16 on GPU — needs ~32GB+ VRAM. Don't pick this on 24GB.
        pipe = StableDiffusion3Pipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
        )
        pipe.to("cuda")
        print("  Full fp16 on CUDA (no quantization).")

    pipe.set_progress_bar_config(disable=True)
    return pipe


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate_batch(
    pipe: StableDiffusion3Pipeline,
    batch: list[dict],
    steps: int,
    guidance: float,
    height: int,
    width: int,
) -> list[Image.Image]:
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
    quantize_t5: bool,
    limit: int | None,
) -> None:
    OUT.mkdir(exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

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

    if not torch.cuda.is_available():
        print("ERROR: SD 3 Medium requires CUDA. Aborting.")
        sys.exit(1)

    dtype = torch.float16
    pipe = load_pipeline(
        low_vram=low_vram,
        quantize_t5=quantize_t5,
        dtype=dtype,
    )

    t_start = time.time()
    n_done = 0
    n_fail = 0

    with tqdm(total=len(todo), unit="img", desc="generating") as pbar:
        for batch in batched(todo, batch_size):
            try:
                images = generate_batch(pipe, batch, steps, guidance, height, width)
            except torch.cuda.OutOfMemoryError:
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
                    help="Enable CPU offload. Use only for <20GB cards.")
    ap.add_argument("--no-quantize", action="store_true",
                    help="Skip 8-bit T5 quantization. Needs ~32GB+ VRAM.")
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
        quantize_t5=not args.no_quantize,
        limit=args.limit,
    )


if __name__ == "__main__":
    sys.exit(main())