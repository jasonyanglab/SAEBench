import os
import json
import pandas as pd

def monitor_info_theory_results(results_dir="eval_results/info_theory"):
    """
    监控并打印信息论评估指标的结果
    """
    if not os.path.exists(results_dir):
        print(f"Directory {results_dir} does not exist yet. Please wait for the job to output results.")
        return

    data = []
    # 遍历结果文件夹中的所有 json 文件
    for filename in os.listdir(results_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    result = json.load(f)
                
                # 按照 eval_output.py 中定义的结构提取指标
                metrics = result.get("eval_result_metrics", {}).get("mean", {})
                
                # 提取层级/ID名称
                sae_id = result.get("sae_lens_id", "Unknown")
                
                data.append({
                    "SAE_Layer": sae_id,
                    "Alive_Ratio": f"{metrics.get('alive_features_ratio', 0):.2%}",
                    "Mean_Entropy": round(metrics.get("mean_shannon_entropy", 0), 4),
                    "Mean_KL_Div": round(metrics.get("mean_kl_divergence", 0), 4),
                    "Mean_FDN": round(metrics.get("mean_fdn", 0), 4)
                })
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    if not data:
        print(f"No valid JSON results found in {results_dir}.")
        return

    # 转换为 DataFrame 并进行格式化对齐
    df = pd.DataFrame(data)
    
    # 按照 SAE_Layer 排序（例如按层数顺序排）
    df = df.sort_values(by="SAE_Layer").reset_index(drop=True)
    
    # 打印表格
    print("\n" + "="*80)
    print(" Information Theory Alignment Evaluation Results ".center(80, "="))
    print("="*80)
    print(df.to_string(index=False, justify='center'))
    print("="*80 + "\n")

if __name__ == "__main__":
    monitor_info_theory_results()