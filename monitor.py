import os
import json
import pandas as pd
from pathlib import Path

def find_val(d, target_key):
    """递归遍历字典/列表，查找并返回目标键的值"""
    if isinstance(d, dict):
        if target_key in d:
            return d[target_key]
        for v in d.values():
            res = find_val(v, target_key)
            if res is not None:
                return res
    elif isinstance(d, list):
        for item in d:
            res = find_val(item, target_key)
            if res is not None:
                return res
    return None

def parse_results(base_dir="eval_results"):
    if not os.path.exists(base_dir):
        print(f"目录 {base_dir} 不存在。")
        return

    data = {}
    base_path = Path(base_dir)

    # 遍历任务目录
    for eval_task_dir in base_path.iterdir():
        if not eval_task_dir.is_dir():
            continue
            
        task_name = eval_task_dir.name
        
        # 遍历 JSON 文件
        for item in eval_task_dir.iterdir():
            json_path = None
            sae_name = None
            
            if item.is_file() and item.name.endswith(".json"):
                json_path = item
                sae_name = item.name.replace("_eval_results.json", "")
            elif item.is_dir():
                json_path = item / "eval_results.json"
                sae_name = item.name
            
            if not json_path or not json_path.exists():
                continue
                
            with open(json_path, 'r', encoding='utf-8') as f:
                try:
                    res = json.load(f)
                except json.JSONDecodeError:
                    continue
            
            if sae_name not in data:
                data[sae_name] = {}
                
            # 利用递归函数提取指标，无视层级嵌套
            if task_name == "core":
                # L0 稀疏度与重构损失
                l0_val = find_val(res, "l0")
                frac_rec = find_val(res, "frac_recovered") or find_val(res, "ce_loss_score")
                data[sae_name]['L0 (Sparsity)'] = round(l0_val, 2) if l0_val is not None else "N/A"
                data[sae_name]['Loss Recovered'] = round(frac_rec, 4) if frac_rec is not None else "N/A"
                
            elif task_name == "absorption":
                score = find_val(res, "mean_absorption_fraction_score")
                data[sae_name]['Absorption Score'] = round(score, 4) if score is not None else "N/A"
                
            elif task_name == "sparse_probing":
                k1_acc = find_val(res, "sae_top_1_test_accuracy")
                data[sae_name]['Probing (k=1)'] = round(k1_acc, 4) if k1_acc is not None else "N/A"
                
            elif task_name in ["scr", "tpp", "scr_and_tpp"]:
                scr_val = find_val(res, "scr_metric_threshold_20")
                tpp_val = find_val(res, "tpp_threshold_20_total_metric")
                score = scr_val if scr_val is not None else tpp_val
                data[sae_name]['SCR/TPP (k=20)'] = round(score, 4) if score is not None else "N/A"
                
            elif task_name == "autointerp":
                score = find_val(res, "autointerp_score")
                data[sae_name]['AutoInterp Score'] = round(score, 4) if score is not None else "N/A"

    if not data:
        print("未找到任何已完成的 JSON 结果文件。")
        return

    df = pd.DataFrame.from_dict(data, orient='index')
    
    # 将缺失值 N/A 替换为 '-' 以保持排版整洁
    df = df.fillna("N/A")
    
    print("\n" + "="*110)
    print(f"当前已生成的评估结果汇总 (共 {len(df)} 个 SAEs)")
    print("="*110)
    print(df.to_string())
    print("="*110 + "\n")

if __name__ == "__main__":
    parse_results()