"""
07_vector_field.py
------------------
Vector-field figure showing SD3's systematic distortion of human SCM ratings.

Design (three layers):
  1. All 66 groups drawn as FAINT gray lines in the background (context)
  2. Per-quadrant SUMMARY arrows in bold quadrant colors:
     each starts at that quadrant's average human position and points to
     its average SD3 position. This is the headline pattern.
  3. The single most-distorted group per quadrant gets a colored arrow
     and a label, placed at the START (human position) where points are
     spread out instead of at the end where everything piles up.

Fiske SCM axes: competence on x, warmth on y.

Input:  group_gaussians_bigg.npz
Output: fig_vectorfield.{pdf,png}
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import procrustes

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "figure.dpi": 120,
})

QUAD_NAMES  = {0: "contempt", 1: "envy", 2: "paternalism", 3: "admiration"}
QUAD_COLORS = {0: "#9c2c2c", 1: "#c47a00", 2: "#2e7d32", 3: "#1565c0"}

# Load
g = np.load("group_gaussians_bigg.npz", allow_pickle=True)
group_keys = g["group_keys"]
mu     = g["mu"]
w_gt   = g["warmth_z_gt"]
c_gt   = g["competence_z_gt"]
quad   = g["quadrant_gt"]

# z-score model coords so they share scale with human z-scores
mu_z_w = (mu[:, 0] - mu[:, 0].mean()) / mu[:, 0].std()
mu_z_c = (mu[:, 1] - mu[:, 1].mean()) / mu[:, 1].std()

# Procrustes-align model coords to human coords. x=competence, y=warmth
human_xy = np.stack([c_gt, w_gt], axis=1)
model_xy = np.stack([mu_z_c, mu_z_w], axis=1)
human_p, model_p, disparity = procrustes(human_xy, model_xy)

# Rescale back to readable z-score units
human_mean = human_xy.mean(0)
scale = np.linalg.norm(human_xy - human_mean, axis=None)
human_plot = human_p * scale + human_mean
model_plot = model_p * scale + human_mean

# -----------------------------------------------------------
# Figure
# -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 8.5))
ax.axhline(0, color="black", lw=0.5, alpha=0.4)
ax.axvline(0, color="black", lw=0.5, alpha=0.4)

# Quadrant corner labels
ax.text(-2.7,  1.85, "Paternalism", color=QUAD_COLORS[2], alpha=1.0, fontsize=12, style="italic", weight="bold")
ax.text( 1.55, 1.85, "Admiration",  color=QUAD_COLORS[3], alpha=1.0, fontsize=12, style="italic", weight="bold")
ax.text(-2.7, -1.95, "Contempt",    color=QUAD_COLORS[0], alpha=1.0, fontsize=12, style="italic", weight="bold")
ax.text( 1.55,-1.95, "Envy",        color=QUAD_COLORS[1], alpha=1.0, fontsize=12, style="italic", weight="bold")

# Layer 1: faint background lines for all 66 groups
for i in range(len(group_keys)):
    hx, hy = human_plot[i]
    mx, my = model_plot[i]
    ax.annotate("",
        xy=(mx, my), xytext=(hx, hy),
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.6, alpha=0.22))

# Layer 2: bold per-quadrant summary arrows
quad_summary = {}
for q in (0, 1, 2, 3):
    mask = quad == q
    if mask.sum() < 2:
        continue
    h_mean = human_plot[mask].mean(0)
    m_mean = model_plot[mask].mean(0)
    quad_summary[q] = (h_mean, m_mean, int(mask.sum()))

for q, (h_mean, m_mean, n) in quad_summary.items():
    color = QUAD_COLORS[q]
    ax.annotate("",
        xy=tuple(m_mean), xytext=tuple(h_mean),
        arrowprops=dict(arrowstyle="-|>", color=color,
                        lw=3.5, alpha=0.95, mutation_scale=22,
                        shrinkA=0, shrinkB=4))
    ax.scatter(*h_mean, s=120, facecolor="white",
               edgecolor=color, linewidths=2.5, zorder=5)
    ax.scatter(*m_mean, s=80, facecolor=color, edgecolor="white",
               linewidths=1.5, zorder=5)

# Layer 3: one labeled exemplar per quadrant (the most distorted group)
residuals = np.linalg.norm(human_plot - model_plot, axis=1)
for q in (0, 1, 2, 3):
    qmask = quad == q
    if qmask.sum() == 0:
        continue
    qidxs = np.where(qmask)[0]
    idx = qidxs[np.argmax(residuals[qidxs])]
    name = str(group_keys[idx])
    hx, hy = human_plot[idx]
    mx, my = model_plot[idx]
    ax.annotate("",
        xy=(mx, my), xytext=(hx, hy),
        arrowprops=dict(arrowstyle="-", color=QUAD_COLORS[q],
                        lw=1.6, alpha=0.65))
    ax.scatter(hx, hy, s=35, facecolor="white",
               edgecolor=QUAD_COLORS[q], linewidths=1.5, zorder=6)
    ax.annotate(name, (hx, hy),
                xytext=(8, 8), textcoords="offset points",
                fontsize=9, color="black", weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="none", alpha=0.75))

ax.set_xlabel("Competence  (z-scored)")
ax.set_ylabel("Warmth  (z-scored)")
ax.set_title("SD3-medium compresses stereotypes toward the origin\n"
             f"Bold arrows: average human → SD3 shift per quadrant.  Procrustes disparity = {disparity:.3f}",
             fontsize=11)
ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
ax.set_aspect("equal")

# Legend
legend = [
    Line2D([], [], marker="o", linestyle="", markerfacecolor="white",
           markeredgecolor="gray", markersize=10, markeredgewidth=2,
           label="Human rating (start)"),
    Line2D([], [], marker=">", linestyle="-", color="gray", lw=2,
           markersize=10, label="SD3 position (end)"),
    Line2D([], [], linestyle="-", color="gray", alpha=0.3, lw=0.8,
           label="Individual groups (background)"),
]
ax.legend(handles=legend, loc="lower left", frameon=False, fontsize=9)

# KL annotation per quadrant summary arrow (small badge near the start)
quad_kl = {
    "contempt":    0.68,
    "envy":        3.51,
    "paternalism": 1.66,
    "admiration":  1.91,
}
offsets = {
    "contempt":    (0.10, -0.25),
    "envy":        (0.10, -0.25),
    "paternalism": (-0.10, 0.10),
    "admiration":  (0.10, 0.10),
}
for q, (h_mean, m_mean, n) in quad_summary.items():
    qname = QUAD_NAMES[q]
    color = QUAD_COLORS[q]
    dx, dy = offsets.get(qname, (0.05, 0.05))
    ha = "right" if dx < 0 else "left"
    va = "top"   if dy < 0 else "bottom"
    ax.text(h_mean[0] + dx, h_mean[1] + dy,
            f"KL = {quad_kl[qname]:.2f}",
            color=color, fontsize=8.5, weight="bold",
            ha=ha, va=va,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor=color, linewidth=0.5, alpha=0.9))

plt.tight_layout()
plt.savefig("plots/fig_vectorfield.pdf", bbox_inches="tight")
plt.savefig("plots/fig_vectorfield.png", bbox_inches="tight", dpi=200)
print("Saved fig_vectorfield.{pdf,png}")
plt.close()