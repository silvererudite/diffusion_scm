"""
08_quadrant_montage.py
----------------------
2x2 quadrant montage showing actual SD3 portraits.

Layout (Fiske SCM convention):
  Top-left:  Paternalism (high W, low C)    e.g. elderly person, childcare worker
  Top-right: Admiration  (high W, high C)   e.g. nurse, doctor
  Bot-left:  Contempt    (low W, low C)     e.g. welfare recipient, garbage collector
  Bot-right: Envy        (low W, high C)    e.g. CEO, lawyer

Each cell shows 2 groups, 3 portraits each.

Portrait selection per group:
  1. Score each of the 100 images by PHOTOREALISM
     (CLIP similarity to "a photograph of a person" minus "a cartoon illustration").
     Drop the bottom 30% as too cartoony / stylized.
  2. Among the photorealistic remainder, pick the 3 closest to the per-group
     CLIP-L mean (most representative real photos), preferring template diversity.

This avoids the issue where some groups (notably CEO) have a mix of real and
cartoony generations — the filter keeps the photo-style ones, which is what
the figure needs.

Inputs:
  embeddings.npz
  HF dataset Shamima/sd3-medium-scm-corpus (streamed)

Output:
  fig_quadrant_montage.{pdf,png}
"""
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from datasets import load_dataset
from transformers import CLIPModel, CLIPProcessor

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "figure.dpi": 120,
})

QUAD_NAMES   = {0: "contempt", 1: "envy", 2: "paternalism", 3: "admiration"}
QUAD_COLORS  = {0: "#9c2c2c", 1: "#c47a00", 2: "#2e7d32", 3: "#1565c0"}

GROUPS_BY_QUAD = {
    "paternalism": ["elderly person", "childcare worker"],
    "admiration":  ["nurse",          "doctor"],
    "contempt":    ["welfare recipient", "garbage collector"],
    "envy":        ["ceo",            "lawyer"],
}

PLURAL_TO_SINGULAR = {
    "rich people": "rich person", "feminists": "feminist",
    "black professionals": "black professional", "elderly people": "elderly person",
    "housewives": "housewife", "welfare recipients": "welfare recipient",
}
def normalize_cat(s):
    s = str(s).strip().lower()
    return PLURAL_TO_SINGULAR.get(s, s)

# -----------------------------------------------------------
# Step 1: Compute a photorealism score for every image
# Uses pre-computed CLIP-L embeddings + a fresh CLIP text-axis encoding.
# Same projection-handling fix as earlier scripts.
# -----------------------------------------------------------
print("Loading embeddings...")
emb = np.load("outputs/embeddings.npz", allow_pickle=True)
clip_emb = emb["clip_emb"]       # (6600, 768) L2-normalized
cats     = emb["categories"]
tidxs    = emb["template_idx"]
global_indices = np.arange(len(cats))

print("Loading CLIP-L text encoder to build photorealism axis...")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
cproc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def clip_text(texts):
    with torch.no_grad():
        tok = cproc(text=texts, return_tensors="pt", padding=True).to(device)
        out = clip.get_text_features(**tok)
        if isinstance(out, torch.Tensor):
            feats = out
        else:
            proj_in = clip.text_projection.weight.shape[1]
            proj_out = clip.text_projection.weight.shape[0]
            p = out.pooler_output
            if p.shape[-1] == proj_in:
                feats = clip.text_projection(p)
            elif p.shape[-1] == proj_out:
                feats = p
            else:
                raise RuntimeError("Unexpected pooler_output dim")
    return feats.cpu().numpy().astype(np.float32)

# Photorealism axis: average of "real photo" prompts minus "cartoon" prompts
real_prompts = [
    "a photograph of a person",
    "a realistic photo portrait",
    "a high-quality photograph of a person",
    "a candid photo of a real person",
]
fake_prompts = [
    "a cartoon illustration of a person",
    "a digital illustration",
    "an animated character",
    "a stylized cartoon portrait",
]
real_vec = clip_text(real_prompts)
real_vec = (real_vec / np.linalg.norm(real_vec, axis=-1, keepdims=True)).mean(0)
fake_vec = clip_text(fake_prompts)
fake_vec = (fake_vec / np.linalg.norm(fake_vec, axis=-1, keepdims=True)).mean(0)
photo_axis = real_vec - fake_vec
photo_axis /= np.linalg.norm(photo_axis)
del clip; torch.cuda.empty_cache()

photo_score = clip_emb @ photo_axis   # (6600,) positive = more photo, negative = more cartoon
print(f"Photorealism score range: {photo_score.min():.3f} to {photo_score.max():.3f}")
print(f"Median: {np.median(photo_score):.3f}")

# -----------------------------------------------------------
# Step 2: Per-group, filter to top-70% photo-realistic then pick 3 most
# representative with template diversity
# -----------------------------------------------------------
all_wanted_groups = sum(GROUPS_BY_QUAD.values(), [])
chosen_indices = {}

for group in all_wanted_groups:
    mask = cats == group
    if mask.sum() == 0:
        print(f"  WARNING: group {group!r} not in embeddings.")
        continue

    g_idxs   = global_indices[mask]
    g_emb    = clip_emb[mask]
    g_temp   = tidxs[mask]
    g_photo  = photo_score[mask]

    # Drop the 30% least photographic
    keep_thresh = np.percentile(g_photo, 30)
    keep_mask = g_photo >= keep_thresh
    g_idxs   = g_idxs[keep_mask]
    g_emb    = g_emb[keep_mask]
    g_temp   = g_temp[keep_mask]
    g_photo  = g_photo[keep_mask]

    # Representativeness: cosine to the photo-filtered group mean
    mean = g_emb.mean(0)
    sims = g_emb @ mean
    order = np.argsort(-sims)

    picks, used_templates = [], set()
    for j in order:
        if g_temp[j] not in used_templates:
            picks.append(int(g_idxs[j]))
            used_templates.add(int(g_temp[j]))
        if len(picks) == 3:
            break
    if len(picks) < 3:
        for j in order:
            if int(g_idxs[j]) not in picks:
                picks.append(int(g_idxs[j]))
            if len(picks) == 3:
                break
    chosen_indices[group] = picks
    print(f"  {group:22s} picked indices {picks}  (kept {keep_mask.sum()}/100 after photo-filter)")

# -----------------------------------------------------------
# Stream the dataset and grab just the rows we need
# -----------------------------------------------------------
print("\nLoading dataset and fetching only the chosen rows...")
needed = set()
for picks in chosen_indices.values():
    needed.update(picks)
print(f"Need {len(needed)} images out of 6600")

ds = load_dataset("Shamima/sd3-medium-scm-corpus", split="train")
# Use .select() for efficient lookup (loads only needed rows)
needed_sorted = sorted(needed)
ds_subset = ds.select(needed_sorted)
# Map global_index -> PIL image
images = {}
for i, row in enumerate(ds_subset):
    idx = needed_sorted[i]
    images[idx] = row["image"].convert("RGB")
print(f"Loaded {len(images)} images")

# -----------------------------------------------------------
# Build the figure
# -----------------------------------------------------------
# Outer 2x2 grid of quadrants; each quadrant has 2 group-rows x 3 portrait-cols
# Total image grid: 4 rows x 6 cols, with gutters between quadrants
QUAD_LAYOUT = [
    ("paternalism", 0, 0),  # row-block 0, col-block 0
    ("admiration",  0, 1),
    ("contempt",    1, 0),
    ("envy",        1, 1),
]
PORTRAITS_PER_GROUP = 3
GROUPS_PER_QUAD     = 2

fig = plt.figure(figsize=(13, 13))
# Use GridSpec for full control. 2 quadrant-rows * 2 group-rows-per-quad = 4 image-rows
# plus 2 separator strips above each quadrant row for the quadrant label.
# Simplest: 2 outer columns x 2 outer rows; each cell is a subgrid.

outer = fig.add_gridspec(2, 2, hspace=0.18, wspace=0.08,
                         left=0.04, right=0.98, top=0.93, bottom=0.04)

for qname, qrow, qcol in QUAD_LAYOUT:
    qid = [k for k, v in QUAD_NAMES.items() if v == qname][0]
    qcolor = QUAD_COLORS[qid]
    inner = outer[qrow, qcol].subgridspec(
        GROUPS_PER_QUAD, PORTRAITS_PER_GROUP, hspace=0.06, wspace=0.04)
    groups = GROUPS_BY_QUAD[qname]

    for g_row, group in enumerate(groups):
        picks = chosen_indices.get(group, [])
        for p_col in range(PORTRAITS_PER_GROUP):
            ax = fig.add_subplot(inner[g_row, p_col])
            if p_col < len(picks):
                ax.imshow(images[picks[p_col]])
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            # Quadrant accent border on the left edge of leftmost image
            if p_col == 0:
                ax.add_patch(Rectangle((0, 0), 6, ax.get_ylim()[0],
                                       transform=ax.transData,
                                       color=qcolor, clip_on=False, zorder=10))
            # Group label below the leftmost portrait
            if p_col == 0:
                ax.set_ylabel(group, fontsize=10.5, rotation=0, ha="right",
                              va="center", labelpad=8, color="black",
                              fontweight="bold")

    # Quadrant header label spanning the inner grid
    # Get bounding box of the inner subgridspec
    bbox = outer[qrow, qcol].get_position(fig)
    fig.text(bbox.x0 + bbox.width / 2, bbox.y1 + 0.005,
             qname.upper(),
             color=qcolor, fontsize=14, fontweight="bold",
             ha="center", va="bottom", style="italic")

fig.suptitle("SD3-medium portraits, organized by SCM quadrant (ground-truth assignment)",
             fontsize=13, y=0.985)

plt.savefig("fig_quadrant_montage.pdf", bbox_inches="tight")
plt.savefig("fig_quadrant_montage.png", bbox_inches="tight", dpi=200)
print("\nSaved fig_quadrant_montage.{pdf,png}")
plt.close()