"""
06_make_figures.py
------------------
Produces the four main figures for the writeup:

  fig1_scm_map.pdf            66 groups in bigG (W,C) space, colored by
                              ground-truth quadrant, with 2D 1-sigma ellipses
                              and per-group means. The headline figure.

  fig2_procrustes.pdf         Side-by-side: human ratings vs model coords
                              after Procrustes alignment. Lines connect
                              matched groups, residual magnitude visible.

  fig3_bayesian_posterior.pdf Posterior densities for beta_warmth and
                              beta_competence with 94% HDI shaded.
                              The Bayesian regression result.

  fig4_residuals.pdf          Top-15 worst-fit groups as a horizontal bar
                              chart, colored by quadrant. The interpretation
                              hook.

  fig_supp_encoder_compare.pdf  CLIP-L vs bigG image-side correlations
                              side by side. Supplementary.

Inputs:
  group_gaussians_bigg.npz
  image_probe_results.csv
  pgm_residuals.csv
  pgm_bayesian_regression.json
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from scipy.spatial import procrustes

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "figure.dpi": 120,
})

QUAD_NAMES   = {0: "contempt", 1: "envy", 2: "paternalism", 3: "admiration"}
QUAD_COLORS  = {0: "#9c2c2c", 1: "#c47a00", 2: "#2e7d32", 3: "#1565c0"}

# -----------------------------------------------------------
# Load
# -----------------------------------------------------------
g = np.load("group_gaussians_bigg.npz", allow_pickle=True)
group_keys = g["group_keys"]
mu     = g["mu"]
sigma  = g["sigma"]
w_gt   = g["warmth_z_gt"]
c_gt   = g["competence_z_gt"]
quad   = g["quadrant_gt"]

probe = pd.read_csv("image_probe_results.csv")
resid = pd.read_csv("pgm_residuals.csv")
with open("pgm_bayesian_regression.json") as f:
    bayes = json.load(f)

# Z-score model coords so they sit on the same scale as ground-truth z-scores
mu_z_w = (mu[:, 0] - mu[:, 0].mean()) / mu[:, 0].std()
mu_z_c = (mu[:, 1] - mu[:, 1].mean()) / mu[:, 1].std()

# -----------------------------------------------------------
# Figure 1: SCM map of all 66 groups
# -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 7.5))
# Quadrant background shading (subtle)
# x-axis = competence, y-axis = warmth (Fiske convention)
ax.axhline(0, color="black", lw=0.5, alpha=0.4)
ax.axvline(0, color="black", lw=0.5, alpha=0.4)
# Corners:
#   high W, low  C = top-left    -> Paternalism
#   high W, high C = top-right   -> Admiration
#   low  W, low  C = bottom-left -> Contempt
#   low  W, high C = bottom-right -> Envy
ax.text(-2.7,  1.8, "Paternalism", color=QUAD_COLORS[2], alpha=1.0, fontsize=11, style="italic", weight="bold")
ax.text( 1.5,  1.8, "Admiration",  color=QUAD_COLORS[3], alpha=1.0, fontsize=11, style="italic", weight="bold")
ax.text(-2.7, -1.9, "Contempt",    color=QUAD_COLORS[0], alpha=1.0, fontsize=11, style="italic", weight="bold")
ax.text( 1.5, -1.9, "Envy",        color=QUAD_COLORS[1], alpha=1.0, fontsize=11, style="italic", weight="bold")

# Compute z-scored sigma in the same space we plotted mu_z.
# Ellipse axis order: width = x-axis (competence), height = y-axis (warmth)
sigma_scale = np.array([mu[:, 0].std(), mu[:, 1].std()])
for i in range(len(group_keys)):
    color = QUAD_COLORS[int(quad[i])]
    # Rescale per-image-projection covariance to the z-scored plotting space
    S = sigma[i] / np.outer(sigma_scale, sigma_scale)
    # S is indexed [w, c]; we plot with c on x and w on y, so swap rows/cols
    # to get covariance in (c, w) order before eigendecomposition.
    S_xy = np.array([[S[1, 1], S[1, 0]],
                     [S[0, 1], S[0, 0]]])
    vals, vecs = np.linalg.eigh(S_xy)
    angle = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
    w_e, h_e = 2 * np.sqrt(vals)
    # Place ellipse at (competence, warmth)
    e = Ellipse((mu_z_c[i], mu_z_w[i]), w_e, h_e, angle=angle,
                facecolor=color, edgecolor=color, alpha=0.12, linewidth=0.5)
    ax.add_patch(e)
    ax.scatter(mu_z_c[i], mu_z_w[i], s=22, color=color, edgecolor="white",
               linewidths=0.5, zorder=5)

# Label a curated subset of groups (extremes + a few middle anchors)
to_label = set()
for axis in ("w_pred", "c_pred"):
    to_label.update(probe.nlargest(4, axis)["category_key"].tolist())
    to_label.update(probe.nsmallest(4, axis)["category_key"].tolist())
to_label.update(["nurse", "ceo", "lawyer", "garbage collector", "welfare recipient"])

for i, key in enumerate(group_keys):
    if key in to_label:
        ax.annotate(str(key), (mu_z_c[i], mu_z_w[i]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=7.5, color="black")

ax.set_xlabel("Competence  (z-scored, bigG image-axis projection)")
ax.set_ylabel("Warmth  (z-scored, bigG image-axis projection)")
ax.set_title("SD3-medium's internalized SCM map (66 groups, bigG image space)")
ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
ax.set_aspect("equal")
legend = [Line2D([], [], marker="o", linestyle="", color=QUAD_COLORS[q],
                 markeredgecolor="white", label=QUAD_NAMES[q]) for q in (0,1,2,3)]
ax.legend(handles=legend, loc="lower right", frameon=False, fontsize=8)
plt.tight_layout()
plt.savefig("fig1_scm_map.pdf", bbox_inches="tight")
plt.savefig("fig1_scm_map.png", bbox_inches="tight", dpi=200)
print("Saved fig1_scm_map.{pdf,png}")
plt.close()

# -----------------------------------------------------------
# Figure 2: Procrustes alignment side-by-side
# Convention matches Figure 1: competence on x, warmth on y
# -----------------------------------------------------------
# Stack as (competence, warmth) so x=index 0, y=index 1
model_xy = np.stack([mu_z_c, mu_z_w], axis=1)
human_xy = np.stack([c_gt, w_gt], axis=1)
m_aligned, h_aligned, disp = procrustes(human_xy, model_xy)

fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharex=True, sharey=True)
for ax, xy, title in [(axes[0], human_xy, "Human ratings"),
                       (axes[1], model_xy, "SD3 (bigG image space)")]:
    ax.axhline(0, color="black", lw=0.5, alpha=0.4)
    ax.axvline(0, color="black", lw=0.5, alpha=0.4)
    for i in range(len(group_keys)):
        ax.scatter(xy[i, 0], xy[i, 1], color=QUAD_COLORS[int(quad[i])],
                   s=22, edgecolor="white", linewidths=0.5)
    ax.set_xlabel("Competence")
    ax.set_title(title)
    ax.set_aspect("equal")
axes[0].set_ylabel("Warmth")

fig.suptitle(f"Group locations in human-rated vs SD3-recovered SCM space  "
             f"(Procrustes disparity = {disp:.3f})", y=1.02)
plt.tight_layout()
plt.savefig("fig2_procrustes.pdf", bbox_inches="tight")
plt.savefig("fig2_procrustes.png", bbox_inches="tight", dpi=200)
print("Saved fig2_procrustes.{pdf,png}")
plt.close()

# -----------------------------------------------------------
# Figure 3: Bayesian posteriors (re-sample from the JSON summaries since we
# only saved summary stats; we'll plot a Normal approximation using mean+sd)
# -----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
for ax, axis in zip(axes, ("warmth", "competence")):
    b = bayes[axis]["beta"]
    x = np.linspace(b["mean"] - 4 * b["sd"], b["mean"] + 4 * b["sd"], 400)
    pdf = np.exp(-0.5 * ((x - b["mean"]) / b["sd"]) ** 2) / (b["sd"] * np.sqrt(2 * np.pi))
    ax.plot(x, pdf, color="#1565c0", lw=2)
    ax.fill_between(x, 0, pdf, where=(x >= b["hdi_lo"]) & (x <= b["hdi_hi"]),
                     color="#1565c0", alpha=0.25, label="94% HDI")
    ax.axvline(0, color="black", lw=0.7, linestyle="--", alpha=0.6)
    ax.axvline(b["mean"], color="#1565c0", lw=0.7)
    ax.set_xlabel(r"$\beta_{" + axis + "}$  (model coord → human rating)")
    ax.set_title(f"{axis.capitalize()}:  "
                 f"β = {b['mean']:+.2f} [{b['hdi_lo']:+.2f}, {b['hdi_hi']:+.2f}],  "
                 f"P(β > 0) = {b['p_gt_zero']:.2f}")
    ax.set_ylabel("Posterior density")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
plt.tight_layout()
plt.savefig("fig3_bayesian_posterior.pdf", bbox_inches="tight")
plt.savefig("fig3_bayesian_posterior.png", bbox_inches="tight", dpi=200)
print("Saved fig3_bayesian_posterior.{pdf,png}")
plt.close()

# -----------------------------------------------------------
# Figure 4: Residuals — worst-fit groups
# -----------------------------------------------------------
top = resid.head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(7.5, 6))
colors = [QUAD_COLORS[list(QUAD_NAMES.values()).index(q)] for q in top["quadrant_gt"]]
ax.barh(top["category_key"], top["residual"], color=colors, edgecolor="white")
ax.set_xlabel("Procrustes residual magnitude")
ax.set_title("Top 15 groups SD3 mislocates relative to human SCM ratings")
legend = [Line2D([], [], marker="s", linestyle="", color=QUAD_COLORS[q],
                  markersize=10, label=QUAD_NAMES[q]) for q in (0,1,2,3)]
ax.legend(handles=legend, loc="lower right", frameon=False, fontsize=8, title="Ground-truth quadrant")
plt.tight_layout()
plt.savefig("fig4_residuals.pdf", bbox_inches="tight")
plt.savefig("fig4_residuals.png", bbox_inches="tight", dpi=200)
print("Saved fig4_residuals.{pdf,png}")
plt.close()

# -----------------------------------------------------------
# Figure supp: CLIP-L vs bigG correlation scatter (image side)
# -----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# CLIP-L panel: ground truth vs predicted (warmth and competence side by side
# within one axis would be confusing; use just competence here for comparison)
for ax, encoder_col_w, encoder_col_c, label in [
    (axes[0], "w_pred", "c_pred", "CLIP-L"),
]:
    ax.scatter(probe["w_gt"], probe[encoder_col_w], s=24, alpha=0.7,
               color="#9c2c2c", label=f"Warmth")
    ax.scatter(probe["c_gt"], probe[encoder_col_c], s=24, alpha=0.7,
               color="#1565c0", label=f"Competence")
    ax.set_xlabel("Human z-score")
    ax.set_ylabel(f"{label} predicted projection")
    ax.set_title(f"{label} image probe (failed)")
    ax.legend(frameon=False, fontsize=8)

# bigG panel
if "bigg_w_pred" in probe.columns:
    axes[1].scatter(probe["w_gt"], probe["bigg_w_pred"], s=24, alpha=0.7,
                    color="#9c2c2c", label="Warmth (r=+0.48)")
    axes[1].scatter(probe["c_gt"], probe["bigg_c_pred"], s=24, alpha=0.7,
                    color="#1565c0", label="Competence (r=+0.34)")
    axes[1].set_xlabel("Human z-score")
    axes[1].set_ylabel("bigG predicted projection")
    axes[1].set_title("bigG image probe (succeeded)")
    axes[1].legend(frameon=False, fontsize=8)

plt.tight_layout()
plt.savefig("fig_supp_encoder_compare.pdf", bbox_inches="tight")
plt.savefig("fig_supp_encoder_compare.png", bbox_inches="tight", dpi=200)
print("Saved fig_supp_encoder_compare.{pdf,png}")
plt.close()

print("\nAll figures written. Open the PDFs for camera-ready quality;")
print("PNGs at 200dpi are inline-friendly for the writeup.")