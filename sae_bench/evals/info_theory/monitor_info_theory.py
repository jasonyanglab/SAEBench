import json
import sys
from pathlib import Path

import pandas as pd


def monitor_info_theory(base_dir: str = "eval_results/info_theory"):
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"目录 {base_dir} 不存在。")
        return

    rows = []

    # 结构: {base_dir}/{model}/{dataset}_{split}_n{samples}_ctx{ctx}/{sae}_eval_results.json
    for json_path in sorted(base_path.rglob("*_eval_results.json")):
        try:
            res = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        # 从路径提取 model / dataset_split
        rel = json_path.relative_to(base_path)
        parts = rel.parts  # e.g. ("gemma-2-2b", "ag_news_train_n2000_ctx128", "xxx_eval_results.json")
        model = parts[0] if len(parts) >= 3 else "unknown"
        ds_split = parts[1] if len(parts) >= 3 else "unknown"

        sae_name = json_path.stem.replace("_eval_results", "")

        # 提取聚合指标
        mean = res.get("eval_result_metrics", {}).get("mean", {})
        kl = mean.get("mean_kl_divergence")
        entropy = mean.get("mean_normalized_entropy")
        alive = mean.get("alive_features_ratio")
        filtered = mean.get("filtered_features_ratio")

        # 提取 eval_config 中的关键参数
        cfg = res.get("eval_config", {})

        rows.append({
            "Model": model,
            "Dataset": ds_split,
            "SAE": sae_name,
            "KL Div": round(kl, 4) if kl is not None else "N/A",
            "Norm Entropy": round(entropy, 4) if entropy is not None else "N/A",
            "Alive%": f"{alive * 100:.1f}" if alive is not None else "N/A",
            "Filtered%": f"{filtered * 100:.1f}" if filtered is not None else "N/A",
            "Samples": cfg.get("num_samples", ""),
            "CtxLen": cfg.get("context_length", ""),
        })

    if not rows:
        print("未找到任何 info_theory 评估结果。")
        return

    df = pd.DataFrame(rows)
    # 按 Model -> Dataset -> KL Div 降序排列
    df = df.sort_values(["Model", "Dataset", "KL Div"], ascending=[True, True, False])
    df = df.reset_index(drop=True)

    print()
    print("=" * 120)
    print(f"  Info Theory 评估结果汇总  (共 {len(df)} 条)")
    print("=" * 120)
    print(df.to_string(index=False))
    print("=" * 120)
    print()


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "eval_results/info_theory"
    monitor_info_theory(base)
