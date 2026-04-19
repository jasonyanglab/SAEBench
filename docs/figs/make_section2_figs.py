"""Generate Section 2 figures: H/KL results analysis.

Produces:
  fig_h_dist_crossdataset.png — 1x3 subplots: H histogram aggregated over 15 SAEs, per dataset
  fig_h_vs_l0_3panel.png      — 1x3 subplots: mean H vs L0 per layer, for each of the 3 datasets
  fig_h_vs_density.png        — scatter H vs density, coloured by layer, pii_noO
"""
import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

BASE = "/Users/yxj/Documents/SAEBench/eval_results/info_theory/gemma-2-2b"
OUT = "/Users/yxj/Documents/SAEBench/docs/figs"

DIRS = {
    "ag_news (C=4, doc)": "ag_news_test_n10000_ctx128",
    "dbpedia14 (C=14, doc)": "dbpedia_14_test_n10000_ctx128",
    "pii_noO (C=25, token)": "pii-masking-300k_validation_n10000_ctx128_noO",
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

ds_order = [
    "ag_news (C=4, doc)",
    "dbpedia14 (C=14, doc)",
    "pii_noO (C=25, token)",
]

# ── Fig 0: 1x3 panel — H histogram aggregated across 15 SAEs per dataset ───────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
panel_colors = {
    "ag_news (C=4, doc)": "#5e81ac",
    "dbpedia14 (C=14, doc)": "#81a1c1",
    "pii_noO (C=25, token)": "#bf616a",
}
bins = np.linspace(0.0, 1.0, 41)
for ax, ds in zip(axes, ds_order):
    h_all = np.concatenate([r["h_filt"] for r in all_data[ds]])
    mean_h = float(np.mean(h_all))
    median_h = float(np.median(h_all))
    frac_lt_03 = float(np.mean(h_all < 0.3))
    ax.hist(h_all, bins=bins, color=panel_colors[ds], alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axvline(mean_h, color="k", lw=1.2, ls="--", label=f"mean = {mean_h:.3f}")
    ax.axvline(median_h, color="k", lw=1.2, ls=":", label=f"median = {median_h:.3f}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("normalized entropy H")
    ax.set_title(f"{ds}\nfrac(H<0.3) = {frac_lt_03:.1%}")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
axes[0].set_ylabel("feature count (15 SAEs combined)")
fig.suptitle(
    "H distribution across datasets — doc-level tasks compress H to [0.7, 1.0]; "
    "token-level pii_noO spreads H across the full [0, 1] range",
    y=1.02,
)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_h_dist_crossdataset.png"), dpi=150, bbox_inches="tight")
plt.close()
print("saved fig_h_dist_crossdataset.png")

# ── Fig A: 1x3 panel — mean H vs L0 per layer, for each dataset ────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
layer_colors = {5: "#1f77b4", 12: "#2ca02c", 19: "#d62728"}
layer_markers = {5: "o", 12: "s", 19: "^"}

for ax, ds in zip(axes, ds_order):
    for layer in [5, 12, 19]:
        rs = sorted([r for r in all_data[ds] if r["layer"] == layer], key=lambda x: x["l0"])
        l0s = [r["l0"] for r in rs]
        mh = [r["mean_h"] for r in rs]
        ax.plot(
            l0s, mh,
            marker=layer_markers[layer], linestyle="-",
            color=layer_colors[layer], label=f"layer {layer}",
            lw=2, ms=9,
        )
    ax.set_xscale("log")
    ax.set_xlabel("L0 (avg active features per token)")
    ax.set_title(ds)
    ax.grid(alpha=0.3)
    ax.set_ylim(0.3, 1.0)
axes[0].set_ylabel("mean H (density-filtered)")
axes[0].legend(loc="lower right")
fig.suptitle("mean H vs L0 across 3 layers, by dataset — layer encodes different concept granularity", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_h_vs_l0_3panel.png"), dpi=150, bbox_inches="tight")
plt.close()
print("saved fig_h_vs_l0_3panel.png")

# ── Fig B: H vs density scatter, pii_noO, coloured by L0 band ─────────────────
def _l0_band_idx(l0):
    if l0 < 25:
        return 0
    elif l0 < 50:
        return 1
    elif l0 < 100:
        return 2
    elif l0 < 200:
        return 3
    else:
        return 4


band_names = [
    "ultra-sparse (<25)",
    "sparse (25–50)",
    "mid (50–100)",
    "dense (100–200)",
    "very-dense (>200)",
]
band_colors = ["#2e1f6b", "#7c3a93", "#c04f76", "#ee7b4c", "#f5c242"]

fig, ax = plt.subplots(figsize=(7.5, 5))
rng = np.random.default_rng(0)
for bi, bname in enumerate(band_names):
    h_all = np.concatenate(
        [r["h_filt"] for r in all_data["pii_noO (C=25, token)"] if _l0_band_idx(r["l0"]) == bi]
    )
    d_all = np.concatenate(
        [r["dens_filt"] for r in all_data["pii_noO (C=25, token)"] if _l0_band_idx(r["l0"]) == bi]
    )
    if len(h_all) > 5000:
        idx = rng.choice(len(h_all), 5000, replace=False)
        h_all, d_all = h_all[idx], d_all[idx]
    ax.scatter(d_all, h_all, s=4, alpha=0.3, color=band_colors[bi], label=bname)

# reference lines: H/KL upper cut (this chapter), P/R lower floor (downstream)
ax.axvline(1e-2, color="k", lw=1.1, ls="--", alpha=0.7)
ax.axvline(1e-3, color="0.35", lw=1.1, ls=":", alpha=0.7)
ax.text(
    1e-2, 0.5, "H/KL upper (ch.2)", rotation=90, ha="right", va="center", fontsize=8,
    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
)
ax.text(
    1e-3, 0.5, "P/R lower (ch.3-4)", rotation=90, ha="right", va="center", fontsize=8,
    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
)

ax.set_xscale("log")
ax.set_xlabel("feature density (log)")
ax.set_ylabel("normalized entropy H")
ax.set_title("H vs density, pii_noO  (15 SAEs, density ≤ 0.01, coloured by L0 band)")
ax.grid(alpha=0.3)
leg = ax.legend(markerscale=3, loc="lower left", fontsize=8, framealpha=0.9)
for lh in leg.legend_handles:
    lh.set_alpha(1.0)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_h_vs_density.png"), dpi=150)
plt.close()
print("saved fig_h_vs_density.png")
