"""
04b_image_side_probe_all_encoders.py
------------------------------------
Image-side SCM probe across all GEOMETRICALLY VALID encoder pairs.

Valid pairs (image encoder lives in same shared space as text axes):
  - CLIP-L image  + CLIP-L text axes
  - bigG   image  + bigG   text axes

NOT included (different/incompatible spaces):
  - CLIP-L image + bigG text axes        (different CLIP spaces)
  - any image    + T5 text axes          (T5 not vision-aligned)
  - DINOv2 image + any text axes         (vision-only, no shared space)

For DINOv2 we use the axis-FREE quadrant classifier instead, since DINOv2
has no compatible text axes by design.

Required inputs on disk:
  embeddings.npz                CLIP-L + DINOv2 image embeddings
  text_artifacts.npz            CLIP-L + bigG (+ optional T5) text axes
  groundtruth_clean.csv

Optional input (used if present):
  bigg_image_embeddings.npz     bigG image embeddings from 02b_embed_images_bigG.py

Outputs:
  image_probe_results.csv       per-group W/C per encoder pair + ground truth
  image_probe_summary.json      correlations, Procrustes stats, quadrant accuracy
  group_gaussians_clipL.npz     per-group mu, sigma in CLIP-L W/C space
  group_gaussians_bigg.npz      same in bigG W/C space (if bigG embeddings present)
"""
import json
import os
import numpy as np
import pandas as pd
from scipy.spatial import procrustes
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

# ============================================================
# Load shared artifacts
# ============================================================
emb = np.load("outputs/embeddings.npz", allow_pickle=True)
clip_img = emb["clip_emb"]
dino_img = emb["dino_emb"]
cats     = emb["categories"]

text = np.load("outputs/text_artifacts.npz", allow_pickle=True)

gt = pd.read_csv("outputs/groundtruth_clean.csv")
img_keys = set(np.unique(cats).tolist())
gt = gt[gt["category_key"].isin(img_keys)].reset_index(drop=True)
gt["quadrant"] = (gt["warmth_z"] > 0).astype(int) * 2 + (gt["competence_z"] > 0).astype(int)
QUAD_NAMES = {0:"contempt", 1:"envy", 2:"paternalism", 3:"admiration"}
group_keys = gt["category_key"].tolist()
print(f"Probing {len(group_keys)} groups\n")

# ============================================================
# Generic axis-projection probe
# ============================================================
def run_axis_probe(name, img_emb, img_cats, v_w, v_c, gt, group_keys, n_perm=2000):
    print("=" * 60)
    print(f"  {name}")
    print("=" * 60)

    means = np.stack([img_emb[img_cats == k].mean(0) for k in group_keys])
    w_pred = means @ v_w
    c_pred = means @ v_c
    coords = np.stack([w_pred, c_pred], axis=1)

    r_w, p_w = pearsonr(w_pred, gt["warmth_z"])
    r_c, p_c = pearsonr(c_pred, gt["competence_z"])
    rho_w, _ = spearmanr(w_pred, gt["warmth_z"])
    rho_c, _ = spearmanr(c_pred, gt["competence_z"])
    print(f"  Warmth      Pearson r = {r_w:+.3f}  (p = {p_w:.2e})   Spearman ρ = {rho_w:+.3f}")
    print(f"  Competence  Pearson r = {r_c:+.3f}  (p = {p_c:.2e})   Spearman ρ = {rho_c:+.3f}")

    df = pd.DataFrame({"category_key": group_keys,
                       "w_pred": w_pred, "c_pred": c_pred,
                       "w_gt": gt["warmth_z"].values, "c_gt": gt["competence_z"].values})
    print(f"\n  Top 5 predicted competence:")
    print(df.nlargest(5, "c_pred")[["category_key","c_pred","c_gt"]].to_string(index=False))
    print(f"  Bottom 5 predicted competence:")
    print(df.nsmallest(5, "c_pred")[["category_key","c_pred","c_gt"]].to_string(index=False))
    print(f"  Top 5 predicted warmth:")
    print(df.nlargest(5, "w_pred")[["category_key","w_pred","w_gt"]].to_string(index=False))
    print(f"  Bottom 5 predicted warmth:")
    print(df.nsmallest(5, "w_pred")[["category_key","w_pred","w_gt"]].to_string(index=False))

    human_xy = np.stack([gt["warmth_z"].values, gt["competence_z"].values], axis=1)
    _, _, disparity = procrustes(human_xy, coords)
    rng = np.random.default_rng(0)
    null = np.empty(n_perm)
    for i in range(n_perm):
        _, _, null[i] = procrustes(human_xy, coords[rng.permutation(len(coords))])
    p_proc = (null <= disparity).mean()
    print(f"\n  Procrustes disparity = {disparity:.4f}  (perm p = {p_proc:.4f})")
    print(f"  Null mean = {null.mean():.4f},  5th pct = {np.percentile(null, 5):.4f}")
    print()

    return {
        "name": name,
        "r_w": float(r_w), "p_w": float(p_w), "rho_w": float(rho_w),
        "r_c": float(r_c), "p_c": float(p_c), "rho_c": float(rho_c),
        "procrustes_disparity": float(disparity),
        "procrustes_perm_p": float(p_proc),
    }, df, coords

# ============================================================
# Per-group 2D Gaussian fitting
# ============================================================
def fit_group_gaussians(img_emb, img_cats, v_w, v_c, group_keys):
    G = len(group_keys)
    mu    = np.zeros((G, 2), dtype=np.float32)
    sigma = np.zeros((G, 2, 2), dtype=np.float32)
    n     = np.zeros(G, dtype=np.int32)
    for i, k in enumerate(group_keys):
        mask = img_cats == k
        imgs = img_emb[mask]
        wc = np.stack([imgs @ v_w, imgs @ v_c], axis=1)
        mu[i] = wc.mean(0)
        sigma[i] = np.cov(wc, rowvar=False)
        n[i] = int(mask.sum())
    return mu, sigma, n

# ============================================================
# Probe A: CLIP-L image + CLIP-L text axes
# ============================================================
results = {}
summary_A, df_A, coords_A = run_axis_probe(
    "Probe A:  CLIP-L image + CLIP-L text axes",
    clip_img, cats,
    text["clipL_v_warmth"], text["clipL_v_competence"],
    gt, group_keys,
)
results["clipL"] = summary_A
mu_L, sig_L, n_L = fit_group_gaussians(
    clip_img, cats, text["clipL_v_warmth"], text["clipL_v_competence"], group_keys)
np.savez("group_gaussians_clipL.npz",
    group_keys=np.array(group_keys), mu=mu_L, sigma=sig_L, n=n_L,
    warmth_z_gt=gt["warmth_z"].values, competence_z_gt=gt["competence_z"].values,
    quadrant_gt=gt["quadrant"].values)

# ============================================================
# Probe B: bigG image + bigG text axes (if embeddings available)
# ============================================================
if os.path.exists("outputs/bigg_image_embeddings.npz"):
    bigg_data = np.load("outputs/bigg_image_embeddings.npz", allow_pickle=True)
    bigg_img = bigg_data["bigg_emb"]
    bigg_cats = bigg_data["categories"]
    summary_B, df_B, coords_B = run_axis_probe(
        "Probe B:  bigG image + bigG text axes",
        bigg_img, bigg_cats,
        text["bigG_v_warmth"], text["bigG_v_competence"],
        gt, group_keys,
    )
    results["bigG"] = summary_B
    mu_G, sig_G, n_G = fit_group_gaussians(
        bigg_img, bigg_cats, text["bigG_v_warmth"], text["bigG_v_competence"], group_keys)
    np.savez("group_gaussians_bigg.npz",
        group_keys=np.array(group_keys), mu=mu_G, sigma=sig_G, n=n_G,
        warmth_z_gt=gt["warmth_z"].values, competence_z_gt=gt["competence_z"].values,
        quadrant_gt=gt["quadrant"].values)
    # Merge into df_A as additional columns for the CSV
    df_A["bigg_w_pred"] = df_B["w_pred"].values
    df_A["bigg_c_pred"] = df_B["c_pred"].values
else:
    print("=" * 60)
    print("  Probe B SKIPPED: bigg_image_embeddings.npz not found")
    print("  Run 02b_embed_images_bigG.py first.")
    print("=" * 60 + "\n")

# ============================================================
# Probe C: DINOv2 axis-free quadrant classifier
# ============================================================
print("=" * 60)
print("  Probe C:  DINOv2 image (axis-free, no text alignment)")
print("=" * 60)

dino_means = np.stack([dino_img[cats == k].mean(0) for k in group_keys])
y = gt["quadrant"].values
unique, counts = np.unique(y, return_counts=True)
print(f"  Quadrant distribution: " + ", ".join(
    f"{QUAD_NAMES[u]}={c}" for u, c in zip(unique, counts)))

chance = max(counts) / len(y)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(LogisticRegression(max_iter=2000, C=1.0),
                          dino_means, y, cv=skf, scoring="accuracy")
print(f"  4-way quadrant accuracy (5-fold CV): {scores.mean():.3f} ± {scores.std():.3f}")
print(f"  Majority baseline: {chance:.3f},  random chance: 0.250")

w_sign = (gt["warmth_z"] > 0).astype(int).values
c_sign = (gt["competence_z"] > 0).astype(int).values
scores_w = cross_val_score(LogisticRegression(max_iter=2000), dino_means, w_sign, cv=skf)
scores_c = cross_val_score(LogisticRegression(max_iter=2000), dino_means, c_sign, cv=skf)
print(f"  Binary warmth-sign accuracy:     {scores_w.mean():.3f} ± {scores_w.std():.3f}")
print(f"  Binary competence-sign accuracy: {scores_c.mean():.3f} ± {scores_c.std():.3f}")

results["dinov2_axis_free"] = {
    "quadrant_accuracy_mean": float(scores.mean()),
    "quadrant_accuracy_std":  float(scores.std()),
    "majority_baseline":      float(chance),
    "warmth_sign_accuracy":   float(scores_w.mean()),
    "competence_sign_accuracy": float(scores_c.mean()),
}

# ============================================================
# Save
# ============================================================
df_A["quadrant_gt"] = gt["quadrant"].values
df_A.to_csv("outputs/image_probe_results.csv", index=False)
with open("outputs/image_probe_summary.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 60)
print("  Summary across encoders:")
print("=" * 60)
for key, s in results.items():
    if "r_w" in s:
        print(f"  {key:8s}  r_W={s['r_w']:+.3f}  r_C={s['r_c']:+.3f}  "
              f"Procrustes p={s['procrustes_perm_p']:.4f}")
    else:
        print(f"  {key:8s}  quadrant acc={s['quadrant_accuracy_mean']:.3f}  "
              f"(baseline {s['majority_baseline']:.3f})")

print("\nSaved image_probe_results.csv, image_probe_summary.json,")
print("       group_gaussians_clipL.npz" + (", group_gaussians_bigg.npz" if "bigG" in results else ""))