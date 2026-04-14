"""Generate Section 5 figure: sparse_probing vs H/KL on ag_news.

Produces:
  fig_sparse_probing_vs_h.png — sae_top_k accuracy vs mean_H scatter, coloured by layer
"""
import json
import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt

SP_DIR = "/Users/yxj/Documents/SAEBench/eval_results/sparse_probing"
IT_DIR = "/Users/yxj/Documents/SAEBench/eval_results/info_theory/gemma-2-2b/ag_news_test_n10000_ctx128"
OUT = "/Users/yxj/Documents/SAEBench/docs/figs"


def key(fn):
    m = re.search(r"layer_(\d+)_width_16k_average_l0_(\d+)", fn)
    return int(m.group(1)), int(m.group(2))


sp = {}
for fp in glob.glob(os.path.join(SP_DIR, "*.json")):
    d = json.load(open(fp))
    m = d["eval_result_metrics"]["sae"]
    sp[key(fp)] = (
        m["sae_top_1_test_accuracy"],
        m["sae_top_5_test_accuracy"],
        m["sae_test_accuracy"],
    )

it = {}
for fp in glob.glob(os.path.join(IT_DIR, "*.json")):
    d = json.load(open(fp))
    m = d["eval_result_metrics"]["mean"]
    it[key(fp)] = (m["mean_normalized_entropy"], m["mean_kl_divergence"])

keys = sorted(set(sp) & set(it))
layers = np.array([k[0] for k in keys])
l0s = np.array([k[1] for k in keys])
top1 = np.array([sp[k][0] for k in keys])
top5 = np.array([sp[k][1] for k in keys])
full = np.array([sp[k][2] for k in keys])
mh = np.array([it[k][0] for k in keys])
mkl = np.array([it[k][1] for k in keys])

layer_colors = {5: "#1f77b4", 12: "#2ca02c", 19: "#d62728"}
layer_markers = {5: "o", 12: "s", 19: "^"}

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

# Left: sae_top1 vs mean H
ax = axes[0]
for L in [5, 12, 19]:
    mask = layers == L
    ax.scatter(
        mh[mask], top1[mask],
        s=90, color=layer_colors[L], marker=layer_markers[L],
        edgecolor="black", linewidth=0.6, label=f"layer {L}",
    )
    # Annotate L0
    for x, y, l in zip(mh[mask], top1[mask], l0s[mask]):
        ax.annotate(f"L0={l}", (x, y), xytext=(4, 3), textcoords="offset points", fontsize=7)
r_pearson = np.corrcoef(mh, top1)[0, 1]
ax.set_xlabel("mean normalized entropy H  (lower = more monosemantic)")
ax.set_ylabel("sparse_probing  sae_top_1 accuracy")
ax.set_title(f"top-1 probe vs mean H   (Pearson r = {r_pearson:+.2f})")
ax.grid(alpha=0.3)
ax.legend(loc="lower right")

# Right: sae_top5 and full vs mean H
ax = axes[1]
for L in [5, 12, 19]:
    mask = layers == L
    ax.scatter(
        mh[mask], top5[mask],
        s=90, color=layer_colors[L], marker=layer_markers[L],
        edgecolor="black", linewidth=0.6, label=f"layer {L} (top-5)",
    )
    ax.scatter(
        mh[mask], full[mask],
        s=55, color=layer_colors[L], marker=layer_markers[L],
        edgecolor="black", linewidth=0.6, alpha=0.35,
    )
r_top5 = np.corrcoef(mh, top5)[0, 1]
r_full = np.corrcoef(mh, full)[0, 1]
ax.set_xlabel("mean normalized entropy H")
ax.set_ylabel("sparse_probing accuracy")
ax.set_title(f"top-5 (solid)  r={r_top5:+.2f}   |   full SAE (faint)  r={r_full:+.2f}")
ax.set_ylim(0.6, 1.0)
ax.axhline(0.95, color="gray", ls="--", lw=1, alpha=0.5)
ax.text(0.805, 0.955, "full-SAE probe ≈ 0.95 (flat)", fontsize=8, color="gray")
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=8)

fig.suptitle(
    "sparse_probing vs H/KL on ag_news — top-k probe rewards denser SAEs while H rewards concentration",
    y=1.02,
)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_sparse_probing_vs_h.png"), dpi=150, bbox_inches="tight")
plt.close()
print("saved fig_sparse_probing_vs_h.png")
