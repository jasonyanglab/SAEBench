"""Generate Section 4.4 figure: L0-tier monotonicity of h_f ampP across k.

Produces:
  fig_l0_tier_monotonicity.png — single panel: h_f ampP vs L0 tier (0..4),
                                   three lines for k=5/10/20, showing that
                                   monotonic descent is cleanest at k=5 and
                                   breaks down (tie at k=10, reversal at k=20)
                                   as k grows.
"""
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

BASE = "/Users/yxj/Documents/SAEBench/eval_results/topk_pr_verification_v9_amp"
OUT = "/Users/yxj/Documents/SAEBench/docs/figs"
K_SET = [5, 10, 20]


def parse(fp):
    m = re.search(r"layer_(\d+).*l0_(\d+)", fp)
    return int(m.group(1)), int(m.group(2))


files = sorted(glob.glob(os.path.join(BASE, "*.json")))
per_layer = defaultdict(list)
for fp in files:
    layer, l0 = parse(fp)
    per_layer[layer].append((l0, fp))

tier_of = {}
for layer, items in per_layer.items():
    items.sort()
    for t, (_, fp) in enumerate(items):
        tier_of[fp] = t

by_tier_k = defaultdict(list)
for fp in files:
    with open(fp) as f:
        d = json.load(f)
    macro = d["macro"]
    tier = tier_of[fp]
    for k in K_SET:
        entry = macro.get(str(k), macro.get(k))
        by_tier_k[(tier, k)].append(entry["h_f_top_amp_precision"])

tiers = list(range(5))
means = {k: np.array([np.mean(by_tier_k[(t, k)]) for t in tiers]) for k in K_SET}

fig, ax = plt.subplots(figsize=(5.6, 4.0))
colors = {5: "#2e7d32", 10: "#ef6c00", 20: "#c62828"}
markers = {5: "o", 10: "s", 20: "D"}
for k in K_SET:
    ax.plot(
        tiers,
        means[k],
        color=colors[k],
        marker=markers[k],
        markersize=7,
        linewidth=2,
        label=f"k = {k}",
    )

ax.set_xticks(tiers)
ax.set_xticklabels([f"tier {t}" for t in tiers])
ax.set_xlabel("L0 tier (0 = sparsest → 4 = densest)")
ax.set_ylabel("h_f amplitude precision (macro over 3 SAEs)")
ax.set_title(
    "L0-tier monotonicity of h_f ampP — cleanest at k=5,\n"
    "ties at k=10 (tier 0/1 ≈ 0.738), reversal at k=20 (tier 0 < tier 1)"
)
ax.grid(alpha=0.3)
ax.legend(loc="upper right", frameon=True)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_l0_tier_monotonicity.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved fig_l0_tier_monotonicity.png")
for k in K_SET:
    print(f"  k={k}: " + ", ".join(f"tier{t}={means[k][t]:.4f}" for t in tiers))
