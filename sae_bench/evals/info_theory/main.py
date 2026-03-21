import argparse
import gc
import os
import time
from dataclasses import asdict
from datetime import datetime
from collections import defaultdict

import einops
import numpy as np
import torch
from scipy.stats import entropy
from sae_lens import SAE
from tqdm import tqdm
from transformer_lens import HookedTransformer

import sae_bench.sae_bench_utils.activation_collection as activation_collection
import sae_bench.sae_bench_utils.general_utils as general_utils
from sae_bench.sae_bench_utils import get_eval_uuid, get_sae_bench_version, get_sae_lens_version
from sae_bench.sae_bench_utils.sae_selection_utils import get_saes_from_regex
from datasets import load_dataset

from sae_bench.evals.info_theory.eval_config import InfoTheoryEvalConfig
from sae_bench.evals.info_theory.eval_output import (
    EVAL_TYPE_ID_INFO_THEORY,
    InfoTheoryEvalOutput,
    InfoTheoryMeanMetrics,
    InfoTheoryMetricCategories,
    InfoTheoryResultDetail,
)

@torch.no_grad()
def get_dataset_activations_agnews(
    model: HookedTransformer,
    config: InfoTheoryEvalConfig,
    layer: int,
    hook_point: str,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    print(f"Loading {config.dataset_name} dataset...")
    # dataset = load_dataset(config.dataset_name, split=f"train[:{config.num_samples}]")
    # 强制固定随机种子打乱，确保完全随机且可复现
    dataset = load_dataset(config.dataset_name, split="train")
    dataset = dataset.shuffle(seed=config.random_seed).select(range(config.num_samples))
    
    texts = dataset['text']
    labels = torch.tensor(dataset['label'], device=device)
    
    tokens = model.tokenizer(
        texts, 
        max_length=config.context_length, 
        truncation=True, 
        padding="max_length", 
        return_tensors="pt"
    )["input_ids"].to(device)
    
    acts_list = []
    for i in tqdm(range(0, len(tokens), config.llm_batch_size), desc="LLM Forward Pass"):
        batch_tokens = tokens[i:i+config.llm_batch_size]
        _, cache = model.run_with_cache(
            batch_tokens,
            names_filter=[hook_point],
            stop_at_layer=layer + 1
        )
        acts_list.append(cache[hook_point].cpu())
        del cache
    
    all_acts_BLD = torch.cat(acts_list, dim=0)
    return all_acts_BLD, labels

# @torch.no_grad()
# def evaluate_features_information_theory(
#     sae_acts_BLF: torch.Tensor,
#     labels_B: torch.Tensor,
#     num_classes: int = 4
# ) -> tuple[dict[str, float], list[InfoTheoryResultDetail]]:
    
#     B, L, F = sae_acts_BLF.shape
#     doc_acts_BF = einops.reduce(sae_acts_BLF, "B L F -> B F", "sum").detach().float().numpy()
#     labels = labels_B.cpu().detach().numpy()
    
#     class_counts = np.bincount(labels, minlength=num_classes)
#     Q = class_counts / np.sum(class_counts)
#     Q = np.clip(Q, 1e-10, 1.0)
    
#     total_tokens = B * L
#     feature_details = []
    
#     for j in range(F):
#         feature_activations = doc_acts_BF[:, j]
#         density = np.count_nonzero(feature_activations) / total_tokens
#         total_activation = np.sum(feature_activations)
        
#         if total_activation <= 1e-5:
#             feature_details.append(InfoTheoryResultDetail(
#                 feature_index=j, density=0.0, shannon_entropy=-1.0, kl_divergence=-1.0, fdn=-1.0
#             ))
#             continue
            
#         P_t_given_f = np.zeros(num_classes)
#         for c in range(num_classes):
#             P_t_given_f[c] = np.sum(feature_activations[labels == c])
            
#         P = P_t_given_f / total_activation
#         P = np.clip(P, 1e-10, 1.0)
        
#         h = entropy(P, base=2)
#         kl_div = entropy(P, qk=Q, base=2)
#         fdn = np.sum(P ** 2)
        
#         feature_details.append(InfoTheoryResultDetail(
#             feature_index=j, density=float(density), shannon_entropy=float(h), kl_divergence=float(kl_div), fdn=float(fdn)
#         ))
        
#     valid_kl = [d.kl_divergence for d in feature_details if d.kl_divergence >= 0]
#     valid_ent = [d.shannon_entropy for d in feature_details if d.shannon_entropy >= 0]
#     valid_fdn = [d.fdn for d in feature_details if d.fdn >= 0]
    
#     mean_metrics = {
#         'mean_kl_divergence': np.mean(valid_kl) if valid_kl else 0.0,
#         'mean_shannon_entropy': np.mean(valid_ent) if valid_ent else 0.0,
#         'mean_fdn': np.mean(valid_fdn) if valid_fdn else 0.0,
#         'alive_features_ratio': len(valid_kl) / F
#     }
    
#     return mean_metrics, feature_details

@torch.no_grad()
def evaluate_features_information_theory(
    doc_acts_BF: np.ndarray,
    labels_np: np.ndarray,
    num_classes: int = 4
) -> tuple[dict[str, float], list[InfoTheoryResultDetail]]:
    
    # 此时传入的已经是降维好的 [Batch, Features] 张量
    B, F = doc_acts_BF.shape
    
    class_counts = np.bincount(labels_np, minlength=num_classes)
    Q = class_counts / np.sum(class_counts)
    Q = np.clip(Q, 1e-10, 1.0)
    
    # 这里的 total_tokens 是为了计算近似 L0，之前 L=128，我们显式传入
    # 因为降维后 L 维度丢失，我们用 B * 128 还原 Token 总数
    total_tokens = B * 128 
    feature_details = []
    
    for j in range(F):
        feature_activations = doc_acts_BF[:, j]
        density = np.count_nonzero(feature_activations) / total_tokens
        total_activation = np.sum(feature_activations)
        
        if total_activation <= 1e-5:
            feature_details.append(InfoTheoryResultDetail(
                feature_index=j, density=0.0, shannon_entropy=-1.0, kl_divergence=-1.0, fdn=-1.0
            ))
            continue
            
        P_t_given_f = np.zeros(num_classes)
        for c in range(num_classes):
            P_t_given_f[c] = np.sum(feature_activations[labels_np == c])
            
        P = P_t_given_f / total_activation
        P = np.clip(P, 1e-10, 1.0)
        
        h = entropy(P, base=2)
        kl_div = entropy(P, qk=Q, base=2)
        fdn = np.sum(P ** 2)
        
        feature_details.append(InfoTheoryResultDetail(
            feature_index=j, density=float(density), shannon_entropy=float(h), kl_divergence=float(kl_div), fdn=float(fdn)
        ))
        
    valid_kl = [d.kl_divergence for d in feature_details if d.kl_divergence >= 0]
    valid_ent = [d.shannon_entropy for d in feature_details if d.shannon_entropy >= 0]
    valid_fdn = [d.fdn for d in feature_details if d.fdn >= 0]
    
    mean_metrics = {
        'mean_kl_divergence': np.mean(valid_kl) if valid_kl else 0.0,
        'mean_shannon_entropy': np.mean(valid_ent) if valid_ent else 0.0,
        'mean_fdn': np.mean(valid_fdn) if valid_fdn else 0.0,
        'alive_features_ratio': len(valid_kl) / F
    }
    
    return mean_metrics, feature_details

# def run_eval(
#     config: InfoTheoryEvalConfig,
#     selected_saes: list[tuple[str, str]],
#     device: str,
#     output_path: str,
#     artifacts_path: str,
#     force_rerun: bool = False
# ):
#     eval_instance_id = get_eval_uuid()
#     sae_lens_version = get_sae_lens_version()
#     sae_bench_commit_hash = get_sae_bench_version()

#     os.makedirs(output_path, exist_ok=True)
#     artifacts_folder = os.path.join(artifacts_path, EVAL_TYPE_ID_INFO_THEORY, config.model_name)
#     os.makedirs(artifacts_folder, exist_ok=True)
    
#     llm_dtype = general_utils.str_to_dtype(config.llm_dtype)
#     model = HookedTransformer.from_pretrained_no_processing(
#         config.model_name, device=device, dtype=llm_dtype
#     )
    
#     acts_path = os.path.join(artifacts_folder, "agnews_acts.pt")
#     if not os.path.exists(acts_path) or force_rerun:
#         dummy_sae_id, dummy_sae, _ = general_utils.load_and_format_sae(selected_saes[0][0], selected_saes[0][1], device)
#         layer = dummy_sae.cfg.hook_layer
#         hook_point = dummy_sae.cfg.hook_name
        
#         all_acts_BLD, labels_B = get_dataset_activations_agnews(model, config, layer, hook_point, device)
#         torch.save({"acts": all_acts_BLD, "labels": labels_B}, acts_path)
#     else:
#         print(f"Loading cached activations from {acts_path}")
#         cache_data = torch.load(acts_path)
#         all_acts_BLD = cache_data["acts"]
#         labels_B = cache_data["labels"]

#     results_dict = {}

#     for sae_release, sae_object_or_id in tqdm(selected_saes, desc="Evaluating SAEs"):
#         sae_id, sae, _ = general_utils.load_and_format_sae(sae_release, sae_object_or_id, device)
#         sae = sae.to(device=device, dtype=llm_dtype)
        
#         sae_result_path = general_utils.get_results_filepath(output_path, sae_release, sae_id)
#         if os.path.exists(sae_result_path) and not force_rerun:
#             continue
            
#         sae_acts_BLF = sae.encode(all_acts_BLD.to(device)).cpu()
#         mean_metrics, feature_details = evaluate_features_information_theory(sae_acts_BLF, labels_B)
        
#         eval_output = InfoTheoryEvalOutput(
#             eval_config=config,
#             eval_id=eval_instance_id,
#             datetime_epoch_millis=int(time.time() * 1000),
#             eval_result_metrics=InfoTheoryMetricCategories(
#                 mean=InfoTheoryMeanMetrics(**mean_metrics)
#             ),
#             eval_result_details=feature_details,
#             sae_bench_commit_hash=sae_bench_commit_hash,
#             sae_lens_id=sae_id,
#             sae_lens_release_id=sae_release,
#             sae_lens_version=sae_lens_version,
#             sae_cfg_dict=sae.cfg.to_dict(),
#         )
        
#         results_dict[f"{sae_release}_{sae_id}"] = asdict(eval_output)
#         eval_output.to_json_file(sae_result_path, indent=2)
            
#         gc.collect()
#         torch.cuda.empty_cache()
    
#     return results_dict

def run_eval(
    config: InfoTheoryEvalConfig,
    selected_saes: list[tuple[str, str]],
    device: str,
    output_path: str,
    artifacts_path: str,
    force_rerun: bool = False
):
    eval_instance_id = get_eval_uuid()
    sae_lens_version = get_sae_lens_version()
    sae_bench_commit_hash = get_sae_bench_version()

    os.makedirs(output_path, exist_ok=True)
    artifacts_folder = os.path.join(artifacts_path, EVAL_TYPE_ID_INFO_THEORY, config.model_name)
    os.makedirs(artifacts_folder, exist_ok=True)
    
    llm_dtype = general_utils.str_to_dtype(config.llm_dtype)
    model = HookedTransformer.from_pretrained_no_processing(
        config.model_name, device=device, dtype=llm_dtype
    )
    
    acts_path = os.path.join(artifacts_folder, "agnews_acts.pt")
    if not os.path.exists(acts_path) or force_rerun:
        dummy_sae_id, dummy_sae, _ = general_utils.load_and_format_sae(selected_saes[0][0], selected_saes[0][1], device)
        layer = dummy_sae.cfg.hook_layer
        hook_point = dummy_sae.cfg.hook_name
        
        all_acts_BLD, labels_B = get_dataset_activations_agnews(model, config, layer, hook_point, device)
        torch.save({"acts": all_acts_BLD, "labels": labels_B}, acts_path)
    else:
        print(f"Loading cached activations from {acts_path}")
        cache_data = torch.load(acts_path)
        all_acts_BLD = cache_data["acts"]
        labels_B = cache_data["labels"]

    results_dict = {}

    for sae_release, sae_object_or_id in tqdm(selected_saes, desc="Evaluating SAEs"):
        sae_id, sae, _ = general_utils.load_and_format_sae(sae_release, sae_object_or_id, device)
        sae = sae.to(device=device, dtype=llm_dtype)
        
        sae_result_path = general_utils.get_results_filepath(output_path, sae_release, sae_id)
        if os.path.exists(sae_result_path) and not force_rerun:
            continue
            
        # ========== 内存优化版本：分批计算 SAE 激活并即时降维 ==========
        doc_acts_list = []
        # 以 config.sae_batch_size 为步长遍历数据，默认通常为 125
        sae_batch_size = getattr(config, 'sae_batch_size', 125) 
        
        with torch.no_grad():
            for i in range(0, len(all_acts_BLD), sae_batch_size):
                batch_acts = all_acts_BLD[i:i+sae_batch_size].to(device)
                
                # 1. 前向传播得到当前批次的 [Batch, Seq_len, Features]
                batch_sae_acts = sae.encode(batch_acts)
                
                # 2. 立即在 Seq_len 维度上累加降维为 [Batch, Features]，极大压缩体积
                batch_doc_acts = einops.reduce(batch_sae_acts, "B L F -> B F", "sum")
                
                # 3. 转回 CPU 存入列表，清空显存
                doc_acts_list.append(batch_doc_acts.detach().float().cpu())
                
                del batch_acts, batch_sae_acts, batch_doc_acts
                
        # 将所有批次拼接，得到最终的降维张量
        doc_acts_BF = torch.cat(doc_acts_list, dim=0).numpy()
        labels_np = labels_B.cpu().numpy()
        
        # 将降维后的数据送入核心算法评估
        mean_metrics, feature_details = evaluate_features_information_theory(doc_acts_BF, labels_np)
        
        eval_output = InfoTheoryEvalOutput(
            eval_config=config,
            eval_id=eval_instance_id,
            datetime_epoch_millis=int(time.time() * 1000),
            eval_result_metrics=InfoTheoryMetricCategories(
                mean=InfoTheoryMeanMetrics(**mean_metrics)
            ),
            eval_result_details=feature_details,
            sae_bench_commit_hash=sae_bench_commit_hash,
            sae_lens_id=sae_id,
            sae_lens_release_id=sae_release,
            sae_lens_version=sae_lens_version,
            sae_cfg_dict=sae.cfg.to_dict(),
        )
        
        results_dict[f"{sae_release}_{sae_id}"] = asdict(eval_output)
        eval_output.to_json_file(sae_result_path, indent=2)
            
        gc.collect()
        torch.cuda.empty_cache()
    
    return results_dict

def create_config_and_selected_saes(args) -> tuple[InfoTheoryEvalConfig, list[tuple[str, str]]]:
    config = InfoTheoryEvalConfig(
        model_name=args.model_name,
        num_samples=args.num_samples,
    )
    if args.llm_batch_size is not None:
        config.llm_batch_size = args.llm_batch_size
    else:
        config.llm_batch_size = activation_collection.LLM_NAME_TO_BATCH_SIZE[config.model_name]
    if args.llm_dtype is not None:
        config.llm_dtype = args.llm_dtype
    else:
        config.llm_dtype = activation_collection.LLM_NAME_TO_DTYPE[config.model_name]
        
    selected_saes = get_saes_from_regex(args.sae_regex_pattern, args.sae_block_pattern)
    assert len(selected_saes) > 0, "No SAEs selected"
    return config, selected_saes

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--sae_regex_pattern", type=str, required=True)
    parser.add_argument("--sae_block_pattern", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--output_folder", type=str, default="eval_results/info_theory")
    parser.add_argument("--artifacts_path", type=str, default="artifacts")
    parser.add_argument("--llm_batch_size", type=int, default=None)
    parser.add_argument("--llm_dtype", type=str, default=None)
    parser.add_argument("--force_rerun", action="store_true")
    return parser

if __name__ == "__main__":
    args = arg_parser().parse_args()
    device = general_utils.setup_environment()
    config, selected_saes = create_config_and_selected_saes(args)
    run_eval(config, selected_saes, device, args.output_folder, args.artifacts_path, args.force_rerun)