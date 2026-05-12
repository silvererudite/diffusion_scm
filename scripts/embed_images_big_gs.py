"""
02b_embed_images_bigG.py
------------------------
Embed all 6,600 SD3 images with the OpenCLIP ViT-bigG/14 IMAGE encoder.
We already have bigG's TEXT axes in text_artifacts.npz; this script gives us
the image side so we can do the geometrically-valid image-side probe.

Inputs:  same images you used for 02_embed_images.py
Output:  bigg_image_embeddings.npz
           bigg_emb:    (N, 1280) float32, L2-normalized
           categories:  (N,) object (normalized category keys)
           template_idx, sample_idx, seed: (N,) int

Usage (mirroring 02_embed_images.py):
  python 02b_embed_images_bigG.py --hf_dataset Shamima/sd3-medium-scm-corpus
  # or
  python 02b_embed_images_bigG.py --image_root /path --manifest manifest.csv
"""
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

PLURAL_TO_SINGULAR = {
    "rich people":"rich person", "feminists":"feminist",
    "black professionals":"black professional", "elderly people":"elderly person",
    "housewives":"housewife", "welfare recipients":"welfare recipient",
}
def normalize_cat(s):
    s = str(s).strip().lower()
    return PLURAL_TO_SINGULAR.get(s, s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_root", type=str, default=None)
    ap.add_argument("--manifest", type=str, default=None)
    ap.add_argument("--hf_dataset", type=str, default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out", type=str, default="outputs/bigg_image_embeddings.npz")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # ---- iterator over (PIL image, metadata) ----
    if args.hf_dataset:
        from datasets import load_dataset
        ds = load_dataset(args.hf_dataset, split="train")
        def iter_rows():
            for row in ds:
                yield {
                    "image": row["image"].convert("RGB"),
                    "category": normalize_cat(row["category"]),
                    "template_idx": row["template_idx"],
                    "sample_idx":   row["sample_idx"],
                    "seed":         row["seed"],
                }
        n_total = len(ds)
    else:
        assert args.manifest and args.image_root, "Need --manifest and --image_root, or --hf_dataset"
        manifest = pd.read_csv(args.manifest)
        root = Path(args.image_root)
        def iter_rows():
            for _, row in manifest.iterrows():
                yield {
                    "image": Image.open(root / row["image_path"]).convert("RGB"),
                    "category": normalize_cat(row["category"]),
                    "template_idx": int(row["template_idx"]),
                    "sample_idx":   int(row["sample_idx"]),
                    "seed":         int(row["seed"]),
                }
        n_total = len(manifest)

    # ---- load bigG ----
    print("Loading OpenCLIP ViT-bigG/14 (image encoder)...")
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-bigG-14", pretrained="laion2b_s39b_b160k", device=args.device)
    model.eval()
    print(f"  Image embedding dim: {model.visual.output_dim}")

    embs, cats, tids, sids, seeds = [], [], [], [], []
    batch_imgs, batch_meta = [], []

    def flush():
        if not batch_imgs:
            return
        # preprocess applies CLIP image normalization + resize/crop to 224
        tensors = torch.stack([preprocess(img) for img in batch_imgs]).to(args.device)
        with torch.no_grad():
            feats = model.encode_image(tensors)              # (B, 1280)
            feats = feats / feats.norm(dim=-1, keepdim=True) # L2 normalize
        embs.append(feats.cpu().numpy().astype(np.float32))
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

    bigg_emb = np.concatenate(embs, axis=0)
    print(f"\nbigG image embeddings shape: {bigg_emb.shape}")
    print(f"Unique categories: {len(set(cats))}")

    np.savez_compressed(args.out,
        bigg_emb=bigg_emb,
        categories=np.array(cats),
        template_idx=np.array(tids, dtype=np.int32),
        sample_idx=np.array(sids, dtype=np.int32),
        seed=np.array(seeds, dtype=np.int64))
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()