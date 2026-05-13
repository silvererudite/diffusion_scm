"""
05_pgm_analyses.py
------------------
The probabilistic-graphical-model layer. Runs three analyses on the
per-group 2D Gaussians in bigG (W, C) space:

  1. Bayesian measurement-error regression
     Model:  for each group g,
       y_g        ~ Normal(mu_g, sigma_g / sqrt(n_g))   (observed model coord)
       human_axis ~ Normal(alpha + beta * mu_g, tau)
     where mu_g is a latent "true" model location with prior N(0, 1).
     This is the actual graphical model: the per-group Gaussian is a
     measurement of a latent group-level mean, and the human rating
     depends linearly on that latent mean.

  2. Per-quadrant KL divergence
     For each of the 4 SCM quadrants, fit one Gaussian to the model's
     predicted (W, C) coords of all groups assigned to that quadrant by
     ground truth, and one Gaussian to the same groups' human ratings.
     Compute symmetric KL between the two. Identifies which quadrant
     SD3 misrepresents most.

  3. Residual analysis
     After Procrustes alignment of model to human coords, compute the
     per-group residual magnitude. Rank top-10 worst-fit groups for
     qualitative interpretation in the writeup.

Inputs:
  group_gaussians_bigg.npz   per-group mu (G,2), sigma (G,2,2), n (G,)

Outputs:
  pgm_bayesian_regression.json   posterior summaries for alpha, beta, tau
  pgm_quadrant_kl.csv            symmetric KL per quadrant
  pgm_residuals.csv              per-group residuals after Procrustes
  pgm_summary.txt                human-readable summary
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import procrustes

# -----------------------------------------------------------
# Load
# -----------------------------------------------------------
g = np.load("group_gaussians_bigg.npz", allow_pickle=True)
group_keys  = g["group_keys"]
mu          = g["mu"]            # (G, 2) model-predicted (w, c) per group
sigma       = g["sigma"]         # (G, 2, 2)
n           = g["n"]             # (G,)
w_gt        = g["warmth_z_gt"]   # (G,)
c_gt        = g["competence_z_gt"]
quadrant    = g["quadrant_gt"]   # (G,) in {0,1,2,3}
QUAD_NAMES  = {0: "contempt", 1: "envy", 2: "paternalism", 3: "admiration"}
G = len(group_keys)
print(f"Loaded {G} groups, mu shape {mu.shape}, sigma shape {sigma.shape}")

# Per-group standard error of the mean for each axis. Since sigma is the
# covariance of the 100 per-image projections within group, sigma/n gives
# the variance of the per-group MEAN. We use sqrt of the diagonal.
sem = np.sqrt(np.stack([sigma[:, 0, 0], sigma[:, 1, 1]], axis=1) / n[:, None])  # (G, 2)
print(f"Mean SEM across groups:  W={sem[:, 0].mean():.4f},  C={sem[:, 1].mean():.4f}")

# -----------------------------------------------------------
# 1. Bayesian measurement-error regression (PyMC)
# -----------------------------------------------------------
print("\n" + "=" * 60)
print("  1. Bayesian measurement-error regression")
print("=" * 60)

import pymc as pm

# Z-score model coords so alpha and beta are on comparable scales to ground truth
mu_z = (mu - mu.mean(0)) / mu.std(0)
sem_z = sem / mu.std(0)  # propagate the scaling to the SEM

# We fit warmth and competence separately, but in the same graphical model
# we treat each group's observed model coordinate as a noisy measurement
# of a latent true coord, and predict the human rating from the latent.
def fit_axis(name, model_obs, model_sem, human_obs):
    with pm.Model() as m:
        # Latent "true" model coordinate for each group
        mu_latent = pm.Normal("mu_latent", 0, 1, shape=G)
        # Measurement model: observed model coord is noisy estimate of latent
        pm.Normal("model_obs", mu=mu_latent, sigma=model_sem,
                  observed=model_obs)
        # Regression of human rating on latent model coord
        alpha = pm.Normal("alpha", 0, 1)
        beta  = pm.Normal("beta", 0, 1)
        tau   = pm.HalfNormal("tau", 1)
        pm.Normal("human", mu=alpha + beta * mu_latent, sigma=tau,
                  observed=human_obs)
        idata = pm.sample(1000, tune=1000, target_accept=0.95,
                          chains=4, progressbar=False, random_seed=0)
    return idata

print("\n  Sampling Warmth model...")
idata_w = fit_axis("warmth", mu_z[:, 0], sem_z[:, 0], w_gt)
print("  Sampling Competence model...")
idata_c = fit_axis("competence", mu_z[:, 1], sem_z[:, 1], c_gt)

def post_summary(idata, var):
    samples = idata.posterior[var].values.flatten()
    return {
        "mean":     float(samples.mean()),
        "sd":       float(samples.std()),
        "hdi_lo":   float(np.percentile(samples, 3)),
        "hdi_hi":   float(np.percentile(samples, 97)),
        "p_gt_zero": float((samples > 0).mean()),
    }

bayes = {
    "warmth": {
        "alpha": post_summary(idata_w, "alpha"),
        "beta":  post_summary(idata_w, "beta"),
        "tau":   post_summary(idata_w, "tau"),
    },
    "competence": {
        "alpha": post_summary(idata_c, "alpha"),
        "beta":  post_summary(idata_c, "beta"),
        "tau":   post_summary(idata_c, "tau"),
    },
}

print("\n  Posterior summaries (94% HDI in brackets):")
for axis in ("warmth", "competence"):
    b = bayes[axis]["beta"]
    print(f"    {axis:11s}  beta = {b['mean']:+.3f} [{b['hdi_lo']:+.3f}, {b['hdi_hi']:+.3f}]"
          f"   P(beta > 0) = {b['p_gt_zero']:.3f}")

with open("pgm_bayesian_regression.json", "w") as f:
    json.dump(bayes, f, indent=2)

# -----------------------------------------------------------
# 2. Per-quadrant KL divergence
# -----------------------------------------------------------
print("\n" + "=" * 60)
print("  2. Per-quadrant KL divergence")
print("=" * 60)

def fit_gaussian_2d(X):
    """Fit (mean, cov) of 2D points. Returns (mu, Sigma)."""
    return X.mean(0), np.cov(X, rowvar=False) + 1e-6 * np.eye(2)

def kl_mvn(m0, S0, m1, S1):
    """KL(N0 || N1) for 2D Gaussians."""
    k = 2
    iS1 = np.linalg.inv(S1)
    diff = m1 - m0
    return 0.5 * (np.trace(iS1 @ S0) + diff @ iS1 @ diff
                  - k + np.log(np.linalg.det(S1) / np.linalg.det(S0)))

# Both spaces need to be on the same scale to compare. Use z-scoring within each.
model_xy = np.stack([(mu[:, 0] - mu[:, 0].mean()) / mu[:, 0].std(),
                     (mu[:, 1] - mu[:, 1].mean()) / mu[:, 1].std()], axis=1)
human_xy = np.stack([w_gt, c_gt], axis=1)  # already z-scored within source

kl_rows = []
for q in (0, 1, 2, 3):
    mask = quadrant == q
    nq = mask.sum()
    if nq < 3:
        kl_rows.append({"quadrant": QUAD_NAMES[q], "n": int(nq),
                        "kl_symm": np.nan, "note": "too few groups"})
        continue
    m_mod, S_mod = fit_gaussian_2d(model_xy[mask])
    m_hum, S_hum = fit_gaussian_2d(human_xy[mask])
    kl_mh = kl_mvn(m_mod, S_mod, m_hum, S_hum)
    kl_hm = kl_mvn(m_hum, S_hum, m_mod, S_mod)
    kl_rows.append({
        "quadrant": QUAD_NAMES[q],
        "n": int(nq),
        "kl_model_to_human": float(kl_mh),
        "kl_human_to_model": float(kl_hm),
        "kl_symm": float(0.5 * (kl_mh + kl_hm)),
        "model_mean_w": float(m_mod[0]), "model_mean_c": float(m_mod[1]),
        "human_mean_w": float(m_hum[0]), "human_mean_c": float(m_hum[1]),
    })

kl_df = pd.DataFrame(kl_rows).sort_values("kl_symm", ascending=False)
print(kl_df.to_string(index=False))
kl_df.to_csv("pgm_quadrant_kl.csv", index=False)

# -----------------------------------------------------------
# 3. Residual analysis
# -----------------------------------------------------------
print("\n" + "=" * 60)
print("  3. Residual analysis (after Procrustes alignment)")
print("=" * 60)

# Procrustes returns aligned versions of both inputs, scaled into a common frame
mat1, mat2, disparity = procrustes(human_xy, model_xy)
# After procrustes, mat1 and mat2 are both centered and scaled to unit norm.
# Per-group residual is the Euclidean distance between aligned points.
residuals = np.linalg.norm(mat1 - mat2, axis=1)

resid_df = pd.DataFrame({
    "category_key": group_keys,
    "residual":     residuals,
    "quadrant_gt":  [QUAD_NAMES[q] for q in quadrant],
    "model_w":      mu[:, 0],   "model_c":      mu[:, 1],
    "human_w":      w_gt,       "human_c":      c_gt,
}).sort_values("residual", ascending=False)

print(f"  Procrustes disparity: {disparity:.4f}")
print(f"\n  10 worst-fit groups (largest residual after alignment):")
print(resid_df.head(10).to_string(index=False))
print(f"\n  10 best-fit groups:")
print(resid_df.tail(10).iloc[::-1].to_string(index=False))

resid_df.to_csv("pgm_residuals.csv", index=False)

# -----------------------------------------------------------
# Save a plain-text summary for the writeup
# -----------------------------------------------------------
summary_lines = []
summary_lines.append("PGM Analyses Summary")
summary_lines.append("=" * 50)
summary_lines.append("")
summary_lines.append("Bayesian measurement-error regression")
summary_lines.append("-" * 50)
for axis in ("warmth", "competence"):
    b = bayes[axis]["beta"]
    a = bayes[axis]["alpha"]
    t = bayes[axis]["tau"]
    summary_lines.append(f"{axis.upper()}:")
    summary_lines.append(f"  beta:  {b['mean']:+.3f}  94% HDI [{b['hdi_lo']:+.3f}, {b['hdi_hi']:+.3f}]  P(>0)={b['p_gt_zero']:.3f}")
    summary_lines.append(f"  alpha: {a['mean']:+.3f}  94% HDI [{a['hdi_lo']:+.3f}, {a['hdi_hi']:+.3f}]")
    summary_lines.append(f"  tau:   {t['mean']:+.3f}  94% HDI [{t['hdi_lo']:+.3f}, {t['hdi_hi']:+.3f}]")
    summary_lines.append("")
summary_lines.append("Per-quadrant KL (symmetric)")
summary_lines.append("-" * 50)
for _, row in kl_df.iterrows():
    if "kl_symm" in row and not np.isnan(row["kl_symm"]):
        summary_lines.append(f"  {row['quadrant']:12s} (n={row['n']:2d}):  KL = {row['kl_symm']:.3f}")
summary_lines.append("")
summary_lines.append(f"Procrustes disparity: {disparity:.4f}")
summary_lines.append("")
summary_lines.append("Top 5 worst-fit groups:")
for _, row in resid_df.head(5).iterrows():
    summary_lines.append(f"  {row['category_key']:25s}  residual={row['residual']:.3f}  ({row['quadrant_gt']})")

Path("pgm_summary.txt").write_text("\n".join(summary_lines))
print("\n" + "=" * 60)
print("Saved:")
print("  pgm_bayesian_regression.json")
print("  pgm_quadrant_kl.csv")
print("  pgm_residuals.csv")
print("  pgm_summary.txt")
print("=" * 60)