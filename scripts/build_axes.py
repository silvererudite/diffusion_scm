"""
03_build_axes.py
----------------
Construct warmth and competence axis vectors from a curated SCM lexicon.

Two encoders, two axis sets:
  - CLIP: text encoder embeds "a photo of a [adjective] person"  -> mean -> L2
  - DINOv2 is vision-only, no text encoder. We derive its axes by averaging
    over the IMAGE embeddings of the highest- and lowest-rated groups in
    the ground truth (a vision-only proxy). This sidesteps text leakage.

Outputs axes.npz with:
  clip_v_warmth, clip_v_competence   : (768,)
  dino_v_warmth, dino_v_competence   : (1024,)

Sanity check: project a few canonical groups and print their (W, C) coords.
"""

import numpy as np
import pandas as pd
import torch
from transformers import CLIPModel, CLIPProcessor

# Lexicon adapted from Nicolas, Bai & Fiske (2021) SADCAT.
# High-loading adjectives, balanced positive/negative isn't needed -- we want
# the AXIS, so use unipolar high-loading items at the positive pole only.
WARMTH_ADJS = [
    "warm", "friendly", "sincere", "trustworthy", "kind", "caring",
    "honest", "good-natured", "tolerant", "sociable", "generous", "fair",
    "compassionate", "approachable",
]
COMPETENCE_ADJS = [
    "competent", "skilled", "intelligent", "capable", "efficient",
    "confident", "knowledgeable", "clever", "independent", "ambitious",
    "successful", "assertive", "professional", "accomplished",
]

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------- CLIP text axes --------
print("Building CLIP text-based axes...")
clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def embed_adj_list(adjs):
    prompts = [f"a photo of a {a} person" for a in adjs]
    with torch.no_grad():
        tok = proc(text=prompts, return_tensors="pt", padding=True).to(device)
        te = clip.get_text_features(**tok)
        if hasattr(te, "text_embeds"):
            te = te.text_embeds
        elif hasattr(te, "pooler_output"):
            te = te.pooler_output
        te = te / te.norm(dim=-1, keepdim=True)
    return te.mean(0).cpu().numpy().astype(np.float32)

clip_v_warmth     = embed_adj_list(WARMTH_ADJS)
clip_v_warmth     /= np.linalg.norm(clip_v_warmth)
clip_v_competence = embed_adj_list(COMPETENCE_ADJS)
clip_v_competence /= np.linalg.norm(clip_v_competence)

print(f"  warmth-competence axis angle: {np.degrees(np.arccos(np.clip(clip_v_warmth @ clip_v_competence, -1, 1))):.1f}°")

# -------- DINOv2 image-based axes --------
# We define DINOv2's warmth axis as: mean(top-warmth-group images) - mean(bottom-warmth-group images)
# Same for competence. This is a vision-only probe with no text leakage.
print("\nBuilding DINOv2 image-based axes from ground-truth extremes...")
emb = np.load("outputs/embeddings.npz", allow_pickle=True)
dino = emb["dino_emb"]
cats = emb["categories"]

gt = pd.read_csv("outputs/groundtruth_clean.csv")

# Top/bottom 8 by warmth_z and competence_z
n_extreme = 8
top_w    = gt.nlargest(n_extreme,  "warmth_z")["category_key"].tolist()
bot_w    = gt.nsmallest(n_extreme, "warmth_z")["category_key"].tolist()
top_c    = gt.nlargest(n_extreme,  "competence_z")["category_key"].tolist()
bot_c    = gt.nsmallest(n_extreme, "competence_z")["category_key"].tolist()

def axis_from_groups(top_keys, bot_keys, emb_arr, cat_arr):
    top_mean = emb_arr[np.isin(cat_arr, top_keys)].mean(0)
    bot_mean = emb_arr[np.isin(cat_arr, bot_keys)].mean(0)
    v = top_mean - bot_mean
    return v / np.linalg.norm(v)

dino_v_warmth     = axis_from_groups(top_w, bot_w, dino, cats)
dino_v_competence = axis_from_groups(top_c, bot_c, dino, cats)

print(f"  Top warmth groups: {top_w}")
print(f"  Bottom warmth groups: {bot_w}")
print(f"  Top competence groups: {top_c}")
print(f"  Bottom competence groups: {bot_c}")
print(f"  DINOv2 warmth-competence axis angle: {np.degrees(np.arccos(np.clip(dino_v_warmth @ dino_v_competence, -1, 1))):.1f}°")

np.savez("axes.npz",
    clip_v_warmth=clip_v_warmth, clip_v_competence=clip_v_competence,
    dino_v_warmth=dino_v_warmth, dino_v_competence=dino_v_competence)

# -------- Sanity check --------
print("\n=== Sanity check: project canonical groups ===")
clip_emb_arr = emb["clip_emb"]
for canon in ["ceo", "homeless person", "elderly person", "nurse", "garbage collector"]:
    mask = cats == canon
    if mask.sum() == 0:
        print(f"  {canon}: NOT FOUND in embeddings")
        continue
    mu_clip = clip_emb_arr[mask].mean(0)
    mu_dino = dino[mask].mean(0)
    print(f"  {canon:20s}  CLIP W={mu_clip @ clip_v_warmth:+.3f} C={mu_clip @ clip_v_competence:+.3f}"
          f"   DINO W={mu_dino @ dino_v_warmth:+.3f} C={mu_dino @ dino_v_competence:+.3f}")

print("\nIf CEO doesn't have high competence or homeless person doesn't have low W&C,")
print("the axes are broken — debug before running the PGM analyses.")