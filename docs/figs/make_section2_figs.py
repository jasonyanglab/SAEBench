"""Generate Section 2 figures: H/KL results analysis.

Produces:
  fig_h_dist_crossdataset.png  — H distribution on ag_news / dbpedia14 / pii_noO (box plot)
  fig_h_vs_l0.png              — mean H vs L0, per-layer lines, pii_noO
  fig_h_vs_density.png         — scatter H vs density, coloured by layer, pii_noO
"""
import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

BASE = "/Users/yxj/Documents/SAEBench/eval_results/info_theory/gemma-2-2b"
OUT = "/Users/yxj/Documents/SAEBench/docs/figs"

DIRS = {
    "ag_news": "ag_news_test_n10000_ctx128",
    "dbpedia14": "dbpedia_14_test_n10000_ctx128",
    "pii_noO": "pii-masking-300k_validation_n10000_ctx128_noO",
}
MAX_DENS = 0.01


def parse_name(fn):
    stem = fn.replace("_eval_results.json", "")
    parts = stem.split("_")
    layer = int(parts[parts.index("layer") + 1])
    l0 = int(parts[parts.index("l0") + 1])
    return layer, l0


def load_all(ds_dir):
    rows = []
    for fp in sorted(glob.glob(os.path.join(BASE, ds_dir, "*.json"))):
        with open(fp) as f:
            d = json.load(f)
        layer, l0 = parse_name(os.path.basename(fp))
        det = d["eval_result_details"]
        h = np.array([x["normalized_entropy"] for x in det])
        kl = np.array([x["kl_divergence"] for x in det])
        dens = np.array([x["density"] for x in det])
        alive = h >= 0
        in_band = alive & (dens <= MAX_DENS)
        rows.append(
            {
                "layer": layer,
                "l0": l0,
                "h_filt": h[in_band],
                "kl_filt": kl[in_band],
                "dens_filt": dens[in_band],
                "mean_h": d["eval_result_metrics"]["mean"]["mean_normalized_entropy"],
                "mean_kl": d["eval_result_metrics"]["mean"]["mean_kl_divergence"],
            }
        )
    return rows


all_data = {ds: load_all(sub) for ds, sub in DIRS.items()}

# ── Fig 1: cross-dataset H distribution (violin) ───────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
data_for_violin = []
labels = []
colors = ["#a6cee3", "#b2df8a", "#fb9a99"]
for ds in ["ag_news", "dbpedia14", "pii_noO"]:
    pooled = np.concatenate([r["h_filt"] for r in all_data[ds]])
    data_for_violin.append(pooled)
    n_classes = {"ag_news": 4, "dbpedia14": 14, "pii_noO": 25}[ds]
    labels.append(f"{ds}\n(C={n_classes})")

parts = ax.violinplot(data_for_violin, showmedians=True, widths=0.8)
for pc, c in zip(parts["bodies"], colors):
    pc.set_facecolor(c)
    pc.set_alpha(0.75)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(labels)
ax.set_ylabel("Normalized entropy  H ∈ [0, 1]")
ax.set_title("H distribution across datasets (15 SAEs pooled, density-filtered)")
ax.set_ylim(-0.05, 1.05)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_h_dist_crossdataset.png"), dpi=150)
plt.close()
print("saved fig_h_dist_crossdataset.png")

# ── Fig 2: mean H vs L0 per layer, pii_noO ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
layer_colors = {5: "#1f77b4", 12: "#2ca02c", 19: "#d62728"}
for layer in [5, 12, 19]:
    rs = sorted([r for r in all_data["pii_noO"] if r["layer"] == layer], key=lambda x: x["l0"])
    l0s = [r["l0"] for r in rs]
    mh = [r["mean_h"] for r in rs]
    mk = [r["mean_kl"] for r in rs]
    axes[0].plot(l0s, mh, "o-", color=layer_colors[layer], label=f"layer {layer}", lw=2, ms=8)
    axes[1].plot(l0s, mk, "o-", color=layer_colors[layer], label=f"layer {layer}", lw=2, ms=8)

for ax, ylab, title in zip(
    axes,
    ["mean H (filtered)", "mean KL (filtered)"],
    ["mean H vs L0  (pii_noO)", "mean KL vs L0  (pii_noO)"],
):
    ax.set_xscale("log")
    ax.set_xlabel("L0 (avg active features per token)")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_h_vs_l0.png"), dpi=150)
plt.close()
print("saved fig_h_vs_l0.png")

# ── Fig 3: H vs density scatter, pii_noO ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 5))
# sample points (too many otherwise)
rng = np.random.default_rng(0)
for layer in [5, 12, 19]:
    h_all = np.concatenate([r["h_filt"] for r in all_data["pii_noO"] if r["layer"] == layer])
    d_all = np.concatenate([r["dens_filt"] for r in all_data["pii_noO"] if r["layer"] == layer])
    if len(h_all) > 8000:
        idx = rng.choice(len(h_all), 8000, replace=False)
        h_all, d_all = h_all[idx], d_all[idx]
    ax.scatter(
        d_all, h_all, s=4, alpha=0.25, color=layer_colors[layer], label=f"layer {layer}"
    )
ax.set_xscale("log")
ax.set_xlabel("feature density (log)")
ax.set_ylabel("normalized entropy H")
ax.set_title("H vs density, pii_noO  (15 SAEs, density ≤ 0.01)")
ax.grid(alpha=0.3)
leg = ax.legend(markerscale=3)
for lh in leg.legend_handles:
    lh.set_alpha(1.0)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_h_vs_density.png"), dpi=150)
plt.close()
print("saved fig_h_vs_density.png")
