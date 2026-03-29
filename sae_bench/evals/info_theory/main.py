import argparse
import gc
import os
import time
from dataclasses import asdict

import einops
import numpy as np
import torch
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
    InfoTheoryEvalOutput,
    InfoTheoryMeanMetrics,
    InfoTheoryMetricCategories,
    InfoTheoryResultDetail,
)


def _get_dataset_short_name(dataset_name: str) -> str:
    """从 dataset_name 提取简短名用于缓存文件命名。
    e.g. 'fancyzhx/ag_news' -> 'ag_news', 'imdb' -> 'imdb'
    """
    return dataset_name.split("/")[-1]


@torch.no_grad()
def get_dataset_activations(
    model: HookedTransformer,
    config: InfoTheoryEvalConfig,
    layer: int,
    hook_point: str,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """加载数据集并提取 LLM 指定层的激活。

    Returns:
        all_acts_BLD: [B, L, D] LLM 激活
        labels_B: [B] 标签张量
        num_classes: 数据集类别数
    """
    print(f"Loading dataset '{config.dataset_name}' split='{config.dataset_split}'...")
    dataset = load_dataset(config.dataset_name, split=config.dataset_split)
    dataset = dataset.shuffle(seed=config.random_seed)  # type: ignore[union-attr]
    actual_samples = min(config.num_samples, len(dataset))  # type: ignore[arg-type]
    if actual_samples < config.num_samples:
        print(f"[WARN] Dataset only has {actual_samples} rows, using all instead of requested {config.num_samples}")
    dataset = dataset.select(range(actual_samples))  # type: ignore[union-attr]

    texts = dataset[config.text_column]
    labels_list = dataset[config.label_column]
    labels = torch.tensor(labels_list, device=device)
    num_classes = int(labels.max().item()) + 1

    print(f"[DEBUG] Dataset loaded: {len(texts)} samples, {num_classes} classes")
    print(f"[DEBUG]   labels shape={labels.shape}, distribution={torch.bincount(labels).tolist()}")
    print(f"[DEBUG]   Sample text[0]: {texts[0][:100]}...")
    print(f"[DEBUG]   Sample labels[:10]: {labels[:10].tolist()}")

    tokenizer_output = model.tokenizer(  # type: ignore[misc]
        texts,
        max_length=config.context_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    tokens = tokenizer_output["input_ids"].to(device)  # type: ignore[index]

    print(f"[DEBUG] Tokens shape={tokens.shape}, dtype={tokens.dtype}")
    print(f"[DEBUG] Sample tokens[0][:20]: {tokens[0][:20].tolist()}")

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
    print(f"[DEBUG] LLM activations: shape={all_acts_BLD.shape}, dtype={all_acts_BLD.dtype}")
    print(f"[DEBUG]   mean={all_acts_BLD.mean().item():.6f}, std={all_acts_BLD.std().item():.6f}")
    print(f"[DEBUG]   sample all_acts_BLD[0, 0, :5]: {all_acts_BLD[0, 0, :5].tolist()}")
    return all_acts_BLD, labels, num_classes


@torch.no_grad()
def encode_sae_batched(
    sae: SAE,
    all_acts_BLD: torch.Tensor,
    sae_batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """分批编码 SAE 激活，返回文档级聚合激活和 token 级密度。

    Returns:
        doc_acts_BF: [B, F] 每个文档各特征的激活总和
        token_density_F: [F] 每个特征在所有 token 上的激活密度 (非零 token 占比)
    """
    doc_acts_list = []
    nonzero_count_list = []
    total_tokens = 0

    print(f"[DEBUG] encode_sae_batched: input shape={all_acts_BLD.shape}, sae_batch_size={sae_batch_size}")

    for i in range(0, len(all_acts_BLD), sae_batch_size):
        batch_acts = all_acts_BLD[i:i+sae_batch_size].to(device)
        batch_sae_acts = sae.encode(batch_acts)  # [batch, L, F]

        if i == 0:
            print(f"[DEBUG] First SAE encode batch: input={batch_acts.shape} -> output={batch_sae_acts.shape}, dtype={batch_sae_acts.dtype}")
            print(f"[DEBUG]   SAE output sample [0,0,:10]: {batch_sae_acts[0, 0, :10].tolist()}")
            nonzero_in_batch = (batch_sae_acts > 0).sum().item()
            total_in_batch = batch_sae_acts.numel()
            print(f"[DEBUG]   Nonzero ratio in first batch: {nonzero_in_batch}/{total_in_batch} = {nonzero_in_batch/total_in_batch:.6f}")

        # 文档级聚合（先做，再用 > 0 统计非零数，避免两者同时占显存）
        batch_doc_acts = einops.reduce(batch_sae_acts, "B L F -> B F", "sum")
        doc_acts_list.append(batch_doc_acts.detach().float().cpu())
        del batch_doc_acts

        # 统计 token 级非零激活数（bool 掩码与 batch_sae_acts 同形，用完立即释放）
        nonzero_count_list.append((batch_sae_acts > 0).sum(dim=(0, 1)).cpu())
        total_tokens += batch_sae_acts.shape[0] * batch_sae_acts.shape[1]

        del batch_acts, batch_sae_acts

    doc_acts_BF = torch.cat(doc_acts_list, dim=0).numpy()
    token_density_F = (torch.stack(nonzero_count_list).sum(dim=0).float() / total_tokens).numpy()

    print(f"[DEBUG] encode_sae_batched results:")
    print(f"[DEBUG]   doc_acts_BF: shape={doc_acts_BF.shape}, dtype={doc_acts_BF.dtype}")
    print(f"[DEBUG]   doc_acts_BF sample [0,:10]: {doc_acts_BF[0, :10].tolist()}")
    print(f"[DEBUG]   doc_acts_BF nonzero ratio: {np.count_nonzero(doc_acts_BF) / doc_acts_BF.size:.6f}")
    print(f"[DEBUG]   token_density_F: shape={token_density_F.shape}, mean={token_density_F.mean():.6f}, max={token_density_F.max():.6f}")
    print(f"[DEBUG]   token_density_F sample [:10]: {token_density_F[:10].tolist()}")
    print(f"[DEBUG]   total_tokens counted: {total_tokens}")

    return doc_acts_BF, token_density_F


def evaluate_features_information_theory(
    doc_acts_BF: np.ndarray,
    labels_np: np.ndarray,
    token_density_F: np.ndarray,
    num_classes: int,
    min_feature_density: float,
    max_feature_density: float,
) -> tuple[dict[str, float], list[InfoTheoryResultDetail]]:
    """向量化计算所有特征的信息论指标。

    Args:
        min_feature_density: 密度下界，低于此值的特征不参与聚合指标
        max_feature_density: 密度上界，高于此值的特征不参与聚合指标
    """
    _B, F = doc_acts_BF.shape
    log2_C = np.log2(num_classes)

    # 计算各类别的先验分布 Q
    class_counts = np.bincount(labels_np, minlength=num_classes).astype(np.float64)
    Q = class_counts / class_counts.sum()
    Q = np.clip(Q, 1e-10, 1.0)

    print(f"[DEBUG] evaluate_features_information_theory:")
    print(f"[DEBUG]   Input: doc_acts_BF={doc_acts_BF.shape}, labels={labels_np.shape}, F={F}, num_classes={num_classes}")
    print(f"[DEBUG]   Class counts: {class_counts.tolist()}, prior Q: {Q.tolist()}")
    print(f"[DEBUG]   Density filter: [{min_feature_density}, {max_feature_density}]")

    # 向量化：按类别聚合激活
    class_acts = np.zeros((num_classes, F), dtype=np.float64)
    for c in range(num_classes):
        mask = labels_np == c
        class_acts[c] = doc_acts_BF[mask].sum(axis=0)

    total_activation = class_acts.sum(axis=0)  # [F]
    alive_mask = total_activation > 1e-5
    num_alive = int(alive_mask.sum())

    print(f"[DEBUG]   Alive features: {num_alive}/{F} ({num_alive/F*100:.1f}%)")

    # 初始化结果数组（死特征填 -1）
    h_norm_all = np.full(F, -1.0)
    kl_all = np.full(F, -1.0)

    if alive_mask.any():
        # P[c, j] = P(class=c | feature=j)
        P = class_acts[:, alive_mask] / total_activation[alive_mask]  # [C, alive_F]
        P = np.clip(P, 1e-10, 1.0)

        print(f"[DEBUG]   P shape={P.shape} (num_classes x alive_features)")
        print(f"[DEBUG]   P[:, 0] (first alive feature): {P[:, 0].tolist()}")

        # Normalized Shannon entropy: H/log2(C), range [0, 1]
        h_alive = -np.sum(P * np.log2(P), axis=0) / log2_C

        # KL divergence: D_KL(P || Q)
        kl_alive = np.sum(P * np.log2(P / Q[:, None]), axis=0)

        h_norm_all[alive_mask] = h_alive
        kl_all[alive_mask] = kl_alive

        print(f"[DEBUG]   H_norm range: [{h_alive.min():.4f}, {h_alive.max():.4f}], mean={h_alive.mean():.4f}")
        print(f"[DEBUG]   KL range: [{kl_alive.min():.4f}, {kl_alive.max():.4f}], mean={kl_alive.mean():.4f}")

    # 构建每个特征的详细结果（全量输出）
    feature_details = [
        InfoTheoryResultDetail(
            feature_index=j,
            density=float(token_density_F[j]),
            normalized_entropy=float(h_norm_all[j]),
            kl_divergence=float(kl_all[j]),
        )
        for j in range(F)
    ]

    # 密度带通滤波：仅通过筛选的特征参与聚合指标
    density_mask = token_density_F <= max_feature_density
    filtered_mask = alive_mask & density_mask
    num_filtered = int(filtered_mask.sum())

    print(f"[DEBUG]   Density filter: {int(density_mask.sum())} features in band")
    print(f"[DEBUG]   Final filtered (alive & in-band): {num_filtered}/{F} ({num_filtered/F*100:.1f}%)")

    mean_metrics = {
        'mean_kl_divergence': float(np.mean(kl_all[filtered_mask])) if num_filtered else 0.0,
        'mean_normalized_entropy': float(np.mean(h_norm_all[filtered_mask])) if num_filtered else 0.0,
        'alive_features_ratio': num_alive / F,
        'filtered_features_ratio': num_filtered / F,
    }

    print(f"[DEBUG]   Final mean_metrics: {mean_metrics}")

    return mean_metrics, feature_details


def _get_acts_cache_path(
    artifacts_folder: str,
    num_samples: int,
    context_length: int,
    layer: int,
    hook_name: str,
) -> str:
    """生成按数据集+split+样本数+上下文长度+层区分的激活缓存路径。
    e.g. artifacts/info_theory/gemma-2-2b/ag_news_train/acts_n2000_ctx128_layer12_hook_resid_post.pt
    """
    safe_hook = hook_name.replace(".", "_")
    return os.path.join(artifacts_folder, f"acts_n{num_samples}_ctx{context_length}_layer{layer}_{safe_hook}.pt")


def run_eval(
    config: InfoTheoryEvalConfig,
    selected_saes: list[tuple[str, SAE]] | list[tuple[str, str]],
    device: str,
    output_path: str,
    artifacts_path: str,
    force_rerun: bool = False
):
    eval_instance_id = get_eval_uuid()
    sae_lens_version = get_sae_lens_version()
    sae_bench_commit_hash = get_sae_bench_version()

    print(f"[DEBUG] run_eval config: model={config.model_name}, dataset={config.dataset_name}, "
          f"split={config.dataset_split}, text_col={config.text_column}, label_col={config.label_column}")
    print(f"[DEBUG]   num_samples={config.num_samples}, context_length={config.context_length}, "
          f"llm_batch_size={config.llm_batch_size}, sae_batch_size={config.sae_batch_size}, llm_dtype={config.llm_dtype}")
    print(f"[DEBUG]   density filter: [{config.min_feature_density}, {config.max_feature_density}]")
    print(f"[DEBUG] output_path={output_path}, artifacts_path={artifacts_path}, force_rerun={force_rerun}")
    print(f"[DEBUG] Total SAEs to evaluate: {len(selected_saes)}")

    # 按 模型/数据集_split 建子文件夹，避免不同运行配置的结果互相覆盖
    ds_short = _get_dataset_short_name(config.dataset_name)
    ds_split_tag = f"{ds_short}_{config.dataset_split}"
    output_path = os.path.join(output_path, config.model_name, ds_split_tag)
    os.makedirs(output_path, exist_ok=True)
    artifacts_path = os.path.join(artifacts_path, config.model_name, ds_split_tag)
    os.makedirs(artifacts_path, exist_ok=True)

    llm_dtype = general_utils.str_to_dtype(config.llm_dtype)
    print(f"[DEBUG] Loading model {config.model_name} with dtype={llm_dtype} to {device}...")
    model = HookedTransformer.from_pretrained_no_processing(
        config.model_name, device=device, dtype=llm_dtype
    )
    print(f"[DEBUG] Model loaded. n_layers={model.cfg.n_layers}, d_model={model.cfg.d_model}")

    results_dict = {}

    # 缓存：同一层的激活只计算一次（本次调用中所有 SAE 应同层）
    cached_layer: int | None = None
    cached_hook: str | None = None
    all_acts_BLD: torch.Tensor | None = None
    labels_B: torch.Tensor | None = None
    num_classes: int = 0

    for sae_release, sae_object_or_id in tqdm(
        selected_saes, desc="Running SAE evaluation"
    ):
        sae_id, sae, _ = general_utils.load_and_format_sae(sae_release, sae_object_or_id, device)  # type: ignore
        sae = sae.to(device=device, dtype=llm_dtype)

        layer = sae.cfg.hook_layer
        hook_name = sae.cfg.hook_name

        print(f"\n[DEBUG] === Evaluating SAE: {sae_release}_{sae_id} ===")
        print(f"[DEBUG] SAE config: layer={layer}, hook={hook_name}, d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")

        sae_result_path = general_utils.get_results_filepath(output_path, sae_release, sae_id)
        print(f"[DEBUG] Result path: {sae_result_path}")

        if os.path.exists(sae_result_path) and not force_rerun:
            print(f"Skipping {sae_release}_{sae_id} (results exist)")
            continue

        # 如果层/hook 变了（或首次），加载或计算激活
        if layer != cached_layer or hook_name != cached_hook:
            acts_path = _get_acts_cache_path(
                artifacts_path, config.num_samples, config.context_length, layer, hook_name,
            )
            print(f"[DEBUG] Activation cache path: {acts_path}")

            if os.path.exists(acts_path) and not force_rerun:
                print(f"Loading cached activations from {acts_path}")
                cache_data = torch.load(acts_path, map_location="cpu", weights_only=False)
                all_acts_BLD = cache_data["acts"]
                labels_B = cache_data["labels"]
                num_classes = int(cache_data["num_classes"])
                print(f"[DEBUG] Loaded: acts={all_acts_BLD.shape}, labels={labels_B.shape}, num_classes={num_classes}")  # type: ignore[union-attr]
            else:
                print(f"Computing LLM activations for layer {layer} ({hook_name})...")
                all_acts_BLD, labels_B, num_classes = get_dataset_activations(
                    model, config, layer, hook_name, device
                )
                torch.save({"acts": all_acts_BLD, "labels": labels_B.cpu(), "num_classes": num_classes}, acts_path)
                print(f"[DEBUG] Saved activations to {acts_path}, size={os.path.getsize(acts_path)/1024/1024:.1f}MB")

            cached_layer = layer
            cached_hook = hook_name

        labels_np = labels_B.cpu().numpy()  # type: ignore[union-attr]

        sae_cfg_dict = sae.cfg.to_dict()  # 在释放 SAE 前保存配置

        doc_acts_BF, token_density_F = encode_sae_batched(
            sae, all_acts_BLD, config.sae_batch_size, device  # type: ignore[arg-type]
        )
        del sae  # 编码完成后立即释放 GPU 显存，避免与下一个 SAE 同时占用
        torch.cuda.empty_cache()

        mean_metrics, feature_details = evaluate_features_information_theory(
            doc_acts_BF, labels_np, token_density_F, num_classes,
            config.min_feature_density, config.max_feature_density,
        )

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
            sae_cfg_dict=sae_cfg_dict,
        )

        results_dict[f"{sae_release}_{sae_id}"] = asdict(eval_output)
        eval_output.to_json_file(sae_result_path, indent=2)
        print(f"[DEBUG] Results saved to {sae_result_path}")

        gc.collect()
        torch.cuda.empty_cache()

    return results_dict


def create_config_and_selected_saes(args) -> tuple[InfoTheoryEvalConfig, list[tuple[str, str]]]:
    config = InfoTheoryEvalConfig(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        text_column=args.text_column,
        label_column=args.label_column,
        num_samples=args.num_samples,
        min_feature_density=args.min_feature_density,
        max_feature_density=args.max_feature_density,
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

    print(f"[DEBUG] Config created: {config}")
    print(f"[DEBUG] Selected {len(selected_saes)} SAE(s):")
    for release, sae_id in selected_saes:
        print(f"[DEBUG]   {release} / {sae_id}")

    return config, selected_saes


def arg_parser():
    parser = argparse.ArgumentParser(description="Run information theory evaluation on SAE features")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--sae_regex_pattern", type=str, required=True)
    parser.add_argument("--sae_block_pattern", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="fancyzhx/ag_news",
                        help="HuggingFace dataset name (e.g. fancyzhx/ag_news, imdb, SetFit/20_newsgroups)")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--text_column", type=str, default="text",
                        help="Name of the text column in the dataset")
    parser.add_argument("--label_column", type=str, default="label",
                        help="Name of the label column in the dataset")
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--min_feature_density", type=float, default=1e-4,
                        help="Lower density threshold for feature filtering")
    parser.add_argument("--max_feature_density", type=float, default=1e-2,
                        help="Upper density threshold for feature filtering")
    parser.add_argument("--output_folder", type=str, default="eval_results/info_theory")
    parser.add_argument("--artifacts_path", type=str, default="artifacts/info_theory")
    parser.add_argument("--llm_batch_size", type=int, default=None)
    parser.add_argument("--llm_dtype", type=str, default=None)
    parser.add_argument("--force_rerun", action="store_true")
    return parser


if __name__ == "__main__":
    args = arg_parser().parse_args()
    device = general_utils.setup_environment()
    print(f"[DEBUG] Device: {device}")
    config, selected_saes = create_config_and_selected_saes(args)
    run_eval(config, selected_saes, device, args.output_folder, args.artifacts_path, args.force_rerun)
