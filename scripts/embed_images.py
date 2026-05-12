"""
02_embed_images.py
------------------
Embed all 6,600 SD3-generated images with BOTH CLIP ViT-L/14 and DINOv2-large.
Produces embeddings.npz with arrays indexed identically to the manifest.

Usage:
  python 02_embed_images.py --image_root /path/to/images --manifest manifest.csv

If you don't have a manifest CSV, this script can also stream from the HF
dataset directly — pass --hf_dataset Shamima/sd3-medium-scm-corpus.

Expected manifest columns: category, prompt, seed, template_idx, sample_idx, image_path
(image_path can be relative to --image_root)

Output (saved as embeddings.npz):
  clip_emb:   (N, 768) float32, L2-normalized
  dino_emb:   (N, 1024) float32, L2-normalized
  categories: (N,) object  -- normalized category_key for joining
  template_idx, sample_idx, seed: (N,) int
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

PLURAL_TO_SINGULAR = {
    "rich people": "rich person", "feminists": "feminist",
    "black professionals": "black professional", "elderly people": "elderly person",
    "housewives": "housewife", "welfare recipients": "welfare recipient",
}
def normalize_cat(s): 
    s = str(s).strip().lower()
    return PLURAL_TO_SINGULAR.get(s, s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_root", type=str, default=None,
                    help="Directory containing images. Required if not using --hf_dataset.")
    ap.add_argument("--manifest", type=str, default=None,
                    help="CSV with one row per image. See module docstring.")
    ap.add_argument("--hf_dataset", type=str, default=None,
                    help="HuggingFace dataset ID (alternative to local files)")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--out", type=str, default="embeddings.npz")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # -------- Load image iterator --------
    if args.hf_dataset:
        from datasets import load_dataset
        ds = load_dataset(args.hf_dataset, split="train")
        def iter_rows():
            for row in ds:
                yield {
                    "image": row["image"].convert("RGB"),
                    "category": normalize_cat(row["category"]),
                    "template_idx": row["template_idx"],
                    "sample_idx": row["sample_idx"],
                    "seed": row["seed"],
                }
        n_total = len(ds)
    else:
        assert args.manifest and args.image_root, "Need --manifest and --image_root, or --hf_dataset"
        manifest = pd.read_csv(args.manifest)
        root = Path(args.image_root)
        def iter_rows():
            for _, row in manifest.iterrows():
                p = root / row["image_path"]
                yield {
                    "image": Image.open(p).convert("RGB"),
                    "category": normalize_cat(row["category"]),
                    "template_idx": int(row["template_idx"]),
                    "sample_idx": int(row["sample_idx"]),
                    "seed": int(row["seed"]),
                }
        n_total = len(manifest)

    print(f"Embedding {n_total} images on {args.device}")

    # -------- Load models --------
    from transformers import CLIPModel, CLIPProcessor, AutoModel, AutoImageProcessor
    print("Loading CLIP ViT-L/14...")
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(args.device).eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    print("Loading DINOv2-large...")
    dino = AutoModel.from_pretrained("facebook/dinov2-large").to(args.device).eval()
    dino_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-large")

    # -------- Embed in batches --------
    clip_embs, dino_embs = [], []
    cats, tids, sids, seeds = [], [], [], []
    
    batch_imgs, batch_meta = [], []

    def flush():
        if not batch_imgs:
            return
        with torch.no_grad():
            # CLIP
            ci = clip_proc(images=batch_imgs, return_tensors="pt").to(args.device)
            ce = clip.get_image_features(**ci)
            # Some transformers versions wrap this in a BaseModelOutput object
            if hasattr(ce, "image_embeds"):
                ce = ce.image_embeds
            elif hasattr(ce, "pooler_output"):
                ce = ce.pooler_output
            ce = ce / ce.norm(dim=-1, keepdim=True)
            clip_embs.append(ce.cpu().numpy().astype(np.float32))
            # DINOv2 (use [CLS] token pooled output)
            di = dino_proc(images=batch_imgs, return_tensors="pt").to(args.device)
            de = dino(**di).last_hidden_state[:, 0]  # CLS token
            de = de / de.norm(dim=-1, keepdim=True)
            dino_embs.append(de.cpu().numpy().astype(np.float32))
        for m in batch_meta:
            cats.append(m["category"])
            tids.append(m["template_idx"])
            sids.append(m["sample_idx"])
            seeds.append(m["seed"])
        batch_imgs.clear()
        batch_meta.clear()

    for row in tqdm(iter_rows(), total=n_total):
        batch_imgs.append(row["image"])
        batch_meta.append(row)
        if len(batch_imgs) >= args.batch_size:
            flush()
    flush()

    clip_emb = np.concatenate(clip_embs, axis=0)
    dino_emb = np.concatenate(dino_embs, axis=0)
    cats = np.array(cats)
    tids = np.array(tids, dtype=np.int32)
    sids = np.array(sids, dtype=np.int32)
    seeds = np.array(seeds, dtype=np.int64)

    print(f"CLIP shape: {clip_emb.shape}, DINOv2 shape: {dino_emb.shape}")
    print(f"Unique categories: {len(np.unique(cats))}")

    np.savez_compressed(args.out,
        clip_emb=clip_emb, dino_emb=dino_emb,
        categories=cats, template_idx=tids,
        sample_idx=sids, seed=seeds)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()