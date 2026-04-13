"""Recompute aggregate info-theory metrics with a different max_feature_density.

The per-feature details (density, normalized_entropy, kl_divergence) are invariant
to the density threshold — only the aggregate `mean` metrics in
`eval_result_metrics` depend on it. So we can recompute them offline from existing
result JSONs without re-running any SAE forward passes.

Usage:
    python recompute_max_density.py --max_feature_density 4e-2
"""

import argparse
import copy
import glob
import json
import os
from collections import defaultdict


def recompute(details: list[dict], max_density: float) -> dict:
    F = len(details)
    alive = [d for d in details if d["normalized_entropy"] >= 0 and d["kl_divergence"] >= 0]
    num_alive = len(alive)

    filtered = [d for d in alive if d["density"] <= max_density]
    num_filtered = len(filtered)

    if num_filtered:
        mean_kl = sum(d["kl_divergence"] for d in filtered) / num_filtered
        mean_h = sum(d["normalized_entropy"] for d in filtered) / num_filtered
    else:
        mean_kl = 0.0
        mean_h = 0.0

    return {
        "mean_kl_divergence": float(mean_kl),
        "mean_normalized_entropy": float(mean_h),
        "alive_features_ratio": num_alive / F,
        "filtered_features_ratio": num_filtered / F,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results_root",
        default="eval_results/info_theory",
    )
    ap.add_argument(
        "--subdirs",
        nargs="+",
        default=[
            "gemma-2-2b/pii-masking-300k_validation_n10000_ctx128_withO",
            "gemma-2-2b/pii-masking-300k_validation_n10000_ctx128_noO",
        ],
        help="Sub-folders under results_root to process (main_token.py outputs).",
    )
    ap.add_argument("--max_feature_density", type=float, default=4e-2)
    ap.add_argument(
        "--out_suffix",
        default=None,
        help="Suffix tag for new files/folder. Defaults to maxd{value}.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite source files in-place instead of writing to a sibling folder.",
    )
    args = ap.parse_args()

    tag = args.out_suffix or f"maxd{args.max_feature_density:.0e}".replace("-0", "-")

    summary_rows = []

    for sub in args.subdirs:
        src_dir = os.path.join(args.results_root, sub)
        if not os.path.isdir(src_dir):
            print(f"[SKIP] {src_dir} does not exist")
            continue

        if args.overwrite:
            dst_dir = src_dir
        else:
            dst_dir = f"{src_dir}_{tag}"
            os.makedirs(dst_dir, exist_ok=True)

        files = sorted(glob.glob(os.path.join(src_dir, "*_eval_results.json")))
        print(f"\n[{sub}] {len(files)} files")

        for fp in files:
            with open(fp) as f:
                data = json.load(f)

            details = data["eval_result_details"]
            old = data["eval_result_metrics"]["mean"]
            new = recompute(details, args.max_feature_density)

            new_data = copy.deepcopy(data)
            new_data["eval_config"]["max_feature_density"] = args.max_feature_density
            new_data["eval_result_metrics"]["mean"] = new

            out_path = os.path.join(dst_dir, os.path.basename(fp))
            with open(out_path, "w") as f:
                json.dump(new_data, f, indent=2)

            name = os.path.basename(fp).replace("_eval_results.json", "")
            summary_rows.append(
                (sub, name, old, new)
            )

    # Print summary table
    print("\n" + "=" * 120)
    print(f"Recomputed with max_feature_density = {args.max_feature_density}")
    print("=" * 120)
    header = f"{'sae':<60} {'KL old':>10} {'KL new':>10} {'H old':>10} {'H new':>10} {'filt old':>10} {'filt new':>10}"
    by_sub = defaultdict(list)
    for sub, name, old, new in summary_rows:
        by_sub[sub].append((name, old, new))

    for sub, rows in by_sub.items():
        print(f"\n--- {sub} ---")
        print(header)
        for name, old, new in rows:
            print(
                f"{name:<60} "
                f"{old['mean_kl_divergence']:>10.4f} {new['mean_kl_divergence']:>10.4f} "
                f"{old['mean_normalized_entropy']:>10.4f} {new['mean_normalized_entropy']:>10.4f} "
                f"{old['filtered_features_ratio']:>10.4f} {new['filtered_features_ratio']:>10.4f}"
            )


if __name__ == "__main__":
    main()
