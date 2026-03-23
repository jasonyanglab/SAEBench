import torch
import numpy as np
import pandas as pd
from scipy.stats import entropy
import einops
from sae_lens import SAE

def verify_top_features(acts_path, sae_id, sae_release, device="cuda", top_k=4):
    print(f"\nLoading cached activations from {acts_path}...")
    cache_data = torch.load(acts_path)
    all_acts_BLD = cache_data["acts"]
    labels_B = cache_data["labels"].cpu().numpy()
    
    print(f"Loading SAE: {sae_release} - {sae_id}...")
    sae, _, _ = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=device)
    sae.eval()

    num_docs = len(all_acts_BLD)
    num_classes = 4
    class_names = ["World", "Sports", "Business", "Sci/Tech"]
    
    # 同时收集幅值和布尔频次
    mags_list = []
    bools_list = []
    batch_size = 125
    
    with torch.no_grad():
        for i in range(0, num_docs, batch_size):
            batch_acts = all_acts_BLD[i:i+batch_size].to(device)
            batch_sae_acts = sae.encode(batch_acts)
            
            # 幅值累加 [Batch, Features]
            batch_doc_mags = einops.reduce(batch_sae_acts, "B L F -> B F", "sum")
            # 布尔频次 [Batch, Features]
            batch_doc_bools = (einops.reduce(batch_sae_acts, "B L F -> B F", "max") > 0).float()
            
            mags_list.append(batch_doc_mags.cpu().numpy())
            bools_list.append(batch_doc_bools.cpu().numpy())
            
            del batch_acts, batch_sae_acts, batch_doc_mags, batch_doc_bools
            
    doc_mags = np.concatenate(mags_list, axis=0)
    doc_bools = np.concatenate(bools_list, axis=0)
    num_features = doc_mags.shape[1]
    
    results = []
    for j in range(num_features):
        acts_bool_j = doc_bools[:, j]
        doc_count = np.sum(acts_bool_j)
        
        # 1. 频次过滤：如果在少于 10 篇文章中激活，视为偶然噪音剔除
        if doc_count < 10:
            continue
            
        acts_mag_j = doc_mags[:, j]
        total_mag = np.sum(acts_mag_j)
        
        if total_mag <= 1e-5:
            continue
            
        class_mags = np.zeros(num_classes)
        class_bools = np.zeros(num_classes)
        
        for c in range(num_classes):
            mask = (labels_B == c)
            class_mags[c] = np.sum(acts_mag_j[mask])
            class_bools[c] = np.sum(acts_bool_j[mask])
            
        # 2. 幅值计算熵：基于能量分布
        P = class_mags / total_mag
        P = np.clip(P, 1e-10, 1.0)
        h = entropy(P, base=2)
        
        dominant_class = int(np.argmax(class_mags))
        
        results.append({
            "feature_idx": j,
            "entropy": h,
            "dominant_class": dominant_class,
            "total_mag": total_mag,
            "class_mags": class_mags,
            "class_bools": class_bools
        })
        
    df = pd.DataFrame(results)
    
    print(f"\n{'='*80}")
    print(f"Top {top_k} Class-Aligned Features for {sae_id}")
    print(f"(Filtered < 10 docs, Entropy & Precision based on Magnitude, Recall based on Frequency)")
    print(f"{'='*80}")
    
    for c in range(num_classes):
        class_total_docs = np.sum(labels_B == c)
        print(f"\n>>> Class [{c}]: {class_names[c]} (Total Docs: {class_total_docs})")
        
        class_df = df[df['dominant_class'] == c].sort_values('entropy').head(top_k)
        
        if class_df.empty:
            print("  No aligned features found.")
            continue
            
        for _, row in class_df.iterrows():
            idx = int(row['feature_idx'])
            h = row['entropy']
            
            # 精确率 (能量纯度)：目标类别的总幅值 / 全局总幅值
            target_mag = row['class_mags'][c]
            precision = target_mag / row['total_mag']
            
            # 召回率 (文档覆盖率)：目标类别中触发该特征的文档数 / 目标类别的总文档数
            target_docs_fired = row['class_bools'][c]
            recall = target_docs_fired / class_total_docs
            
            print(f"  Feature {idx:5d} | Entropy: {h:.4f} | Mag_Precision: {precision:.2%} | Doc_Recall: {recall:.2%}")

if __name__ == "__main__":
    acts_path = "/scratch/project_2005865/myj_SAE/project/SAEBench/artifacts/info_theory/gemma-2-2b/agnews_acts.pt"
    verify_top_features(acts_path, sae_id="layer_12/width_16k/average_l0_82", sae_release="gemma-scope-2b-pt-res")