import torch
import numpy as np
import pandas as pd
from scipy.stats import entropy
import einops
import os
from dataclasses import dataclass, asdict
from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE

# ==========================================
# 1. Config 模块
# ==========================================
@dataclass
class MultiLayerVerifyConfig:
    model_name: str = "gemma-2-2b"
    sae_release: str = "gemma-scope-2b-pt-res"
    layers: tuple = (5, 12, 19)
    l0_target: str = "average_l0_82"
    top_k: int = 20
    min_recall: float = 0.05
    num_samples: int = 10000
    max_seq_len: int = 128
    llm_batch_size: int = 32
    sae_batch_size: int = 125
    device: str = "cuda"
    output_dir: str = "eval_results/info_theory/gemma-2-2b"
    num_classes: int = 4
    class_names: tuple = ("World", "Sports", "Business", "Sci/Tech")

# ==========================================
# 2. Output 模块
# ==========================================
class MultiLayerOutput:
    def __init__(self, config: MultiLayerVerifyConfig):
        self.config = config
        self.summary_results = []
        os.makedirs(self.config.output_dir, exist_ok=True)
        self.csv_path = os.path.join(self.config.output_dir, f"multi_layer_summary_top{self.config.top_k}.csv")

    def add_layer_results(self, layer_summary: list[dict]):
        self.summary_results.extend(layer_summary)

    def save_and_print(self):
        summary_df = pd.DataFrame(self.summary_results)
        print("\n" + "="*85)
        print(f" Multi-Layer Macro-Concept Alignment (Top-{self.config.top_k}, Recall >= {self.config.min_recall*100}%) ")
        print("="*85)
        print(summary_df.to_string(index=False, float_format="%.4f"))
        print("="*85 + "\n")
        
        summary_df.to_csv(self.csv_path, index=False, float_format="%.4f")
        print(f"Results successfully saved to: {self.csv_path}")

# ==========================================
# 3. 核心计算逻辑
# ==========================================
def run_eval(config: MultiLayerVerifyConfig):
    print(f"Loading Base Model ({config.model_name}) and Dataset...")
    model = HookedTransformer.from_pretrained_no_processing(config.model_name, device=config.device, dtype=torch.bfloat16)
    
    # 强制设置 pad_token 避免 tokenize 报错
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
        
    dataset = load_dataset("fancyzhx/ag_news", split="train")
    dataset = dataset.shuffle(seed=42).select(range(config.num_samples))
    texts = dataset["text"]
    labels_np = np.array(dataset["label"])
    
    output_handler = MultiLayerOutput(config)

    for layer in config.layers:
        print(f"\n{'-'*50}")
        print(f"Processing Layer {layer}...")
        hook_name = f"blocks.{layer}.hook_resid_post"
        
        # 3.1 提取并缓存基座模型激活值 (修复 Tokenizer 对齐问题)
        print("  Extracting LLM activations with strict padding...")
        all_acts = []
        with torch.no_grad():
            for i in range(0, config.num_samples, config.llm_batch_size):
                batch_texts = texts[i:i+config.llm_batch_size]
                
                # 显式 Tokenize，强制统一 Seq_Len 
                tokens = model.tokenizer(
                    batch_texts, 
                    padding="max_length", 
                    truncation=True, 
                    max_length=config.max_seq_len, 
                    return_tensors="pt"
                )["input_ids"].to(config.device)
                
                _, cache = model.run_with_cache(tokens, names_filter=hook_name)
                all_acts.append(cache[hook_name].cpu())
                del cache
                
        all_acts_BLD = torch.cat(all_acts, dim=0) # [Docs, Seq_Len, D_model]
        
        # 3.2 加载 SAE
        sae_id = f"layer_{layer}/width_16k/{config.l0_target}"
        print(f"  Loading SAE: {sae_id}...")
        sae_result = SAE.from_pretrained(release=config.sae_release, sae_id=sae_id, device=config.device)
        sae = sae_result[0] if isinstance(sae_result, tuple) else sae_result
        sae.eval()
        
        # 3.3 提取 SAE 特征的幅值与布尔频次
        print("  Calculating Feature Magnitudes and Frequencies...")
        mags_list, bools_list = [], []
        
        with torch.no_grad():
            for i in range(0, config.num_samples, config.sae_batch_size):
                batch_acts = all_acts_BLD[i:i+config.sae_batch_size].to(config.device)
                batch_sae_acts = sae.encode(batch_acts)
                
                mags_list.append(einops.reduce(batch_sae_acts, "B L F -> B F", "sum").cpu().numpy())
                bools_list.append((einops.reduce(batch_sae_acts, "B L F -> B F", "max") > 0).float().cpu().numpy())
                del batch_acts, batch_sae_acts
                
        doc_mags = np.concatenate(mags_list, axis=0)
        doc_bools = np.concatenate(bools_list, axis=0)
        num_features = doc_mags.shape[1]
        
        # 3.4 双重过滤与信息论指标计算
        features_data = []
        for j in range(num_features):
            acts_bool_j = doc_bools[:, j]
            if np.sum(acts_bool_j) < 10: # 全局频次过滤
                continue
                
            acts_mag_j = doc_mags[:, j]
            total_mag = np.sum(acts_mag_j)
            if total_mag <= 1e-5:
                continue
                
            class_mags = np.zeros(config.num_classes)
            class_bools = np.zeros(config.num_classes)
            for c in range(config.num_classes):
                mask = (labels_np == c)
                class_mags[c] = np.sum(acts_mag_j[mask])
                class_bools[c] = np.sum(acts_bool_j[mask])
                
            P = class_mags / total_mag
            P = np.clip(P, 1e-10, 1.0)
            h = entropy(P, base=2)
            
            dominant_class = int(np.argmax(class_mags))
            precision = class_mags[dominant_class] / total_mag
            recall = class_bools[dominant_class] / np.sum(labels_np == dominant_class)
            
            features_data.append({
                "layer": layer,
                "feature_idx": j,
                "entropy": h,
                "dominant_class": dominant_class,
                "precision": precision,
                "recall": recall
            })
            
        df = pd.DataFrame(features_data)
        layer_summary = []
        
        # 3.5 提取宏观类别 Top-K
        for c in range(config.num_classes):
            class_df = df[(df['dominant_class'] == c) & (df['recall'] >= config.min_recall)]
            top_k_df = class_df.sort_values('entropy').head(config.top_k)
            
            if not top_k_df.empty:
                layer_summary.append({
                    "Layer": layer,
                    "Class": config.class_names[c],
                    "Features_Used": len(top_k_df),
                    "Mean_Entropy": top_k_df['entropy'].mean(),
                    "Mean_Precision": top_k_df['precision'].mean(),
                    "Mean_Recall": top_k_df['recall'].mean()
                })
                
        output_handler.add_layer_results(layer_summary)

    # 4. 执行输出存储
    output_handler.save_and_print()

# ==========================================
# 4. Main 入口
# ==========================================
if __name__ == "__main__":
    cfg = MultiLayerVerifyConfig()
    run_eval(cfg)