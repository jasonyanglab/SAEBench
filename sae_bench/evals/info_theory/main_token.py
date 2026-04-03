"""Token-level information theory evaluation for SAE features.

For datasets with token-level labels (e.g., PII NER), this evaluates
whether individual SAE features align with fine-grained token concepts
rather than coarse document-level categories.

Key difference from main.py (document-level):
- Labels are per-token, not per-document
- No document-level activation aggregation
- Character-level span labels are aligned to model tokens via offset_mapping
- Streaming class_acts accumulation to avoid O(B*L*F) memory blowout
"""

import argparse
import gc
import os
import time
from dataclasses import asdict

import numpy as np
import torch
from sae_lens import SAE
from tqdm import tqdm
from transformer_lens import HookedTransformer
from datasets import Dataset, load_dataset

import sae_bench.sae_bench_utils.activation_collection as activation_collection
import sae_bench.sae_bench_utils.general_utils as general_utils
from sae_bench.sae_bench_utils import get_eval_uuid, get_sae_bench_version, get_sae_lens_version
from sae_bench.sae_bench_utils.sae_selection_utils import get_saes_from_regex

from sae_bench.evals.info_theory.eval_config import InfoTheoryEvalConfig
from sae_bench.evals.info_theory.eval_output import (
    InfoTheoryEvalOutput,
    InfoTheoryMeanMetrics,
    InfoTheoryMetricCategories,
    InfoTheoryResultDetail,
)

# ── Label mapping ──────────────────────────────────────────────────────────────

LABEL_MERGE_MAP = {
    "GIVENNAME1": "GIVENNAME",
    "GIVENNAME2": "GIVENNAME",
    "LASTNAME1": "LASTNAME",
    "LASTNAME2": "LASTNAME",
    "LASTNAME3": "LASTNAME",
}

# 25 entity types after merging name variants (scanned from full train set)
PII_ENTITY_TYPES = sorted([
    "BOD", "BUILDING", "CARDISSUER", "CITY", "COUNTRY", "DATE",
    "DRIVERLICENSE", "EMAIL", "GEOCOORD", "GIVENNAME", "IDCARD", "IP",
    "LASTNAME", "PASS", "PASSPORT", "POSTCODE", "SECADDRESS", "SEX",
    "SOCIALNUMBER", "STATE", "STREET", "TEL", "TIME", "TITLE", "USERNAME",
])

IGNORE_LABEL_ID = -1


def build_label2id(include_non_entity: bool = True) -> dict[str, int]:
    """Build label-to-ID mapping for PII entity types (after merging variants)."""
    label2id: dict[str, int] = {}
    if include_non_entity:
        label2id["O"] = 0
        for i, et in enumerate(PII_ENTITY_TYPES):
            label2id[et] = i + 1
    else:
        for i, et in enumerate(PII_ENTITY_TYPES):
            label2id[et] = i
    return label2id


def align_spans_to_tokens(
    offset_mapping: list[tuple[int, int]],
    spans: list[dict],
    label2id: dict[str, int],
    include_non_entity: bool = True,
) -> list[int]:
    """Align character-level span labels to model tokens via offset overlap.

    Args:
        offset_mapping: [(char_start, char_end), ...] from tokenizer
        spans: [{'start': int, 'end': int, 'label': str}, ...] from privacy_mask
        label2id: mapping from label string to integer ID
        include_non_entity: if False, non-entity tokens get IGNORE_LABEL_ID

    Returns:
        List of label IDs per token. IGNORE_LABEL_ID for PAD/special/excluded tokens.
    """
    token_labels = []
    for tok_start, tok_end in offset_mapping:
        if tok_start == tok_end:  # special token or padding
            token_labels.append(IGNORE_LABEL_ID)
            continue

        label_str = "O"
        for span in spans:
            if tok_start < span["end"] and tok_end > span["start"]:
                raw = span["label"]
                label_str = LABEL_MERGE_MAP.get(raw, raw)
                break

        if label_str == "O" and not include_non_entity:
            token_labels.append(IGNORE_LABEL_ID)
        elif label_str in label2id:
            token_labels.append(label2id[label_str])
        else:
            token_labels.append(IGNORE_LABEL_ID)

    return token_labels


# ── Data loading & activation extraction ───────────────────────────────────────

@torch.no_grad()
def get_token_level_activations(
    model: HookedTransformer,
    config: InfoTheoryEvalConfig,
    layer: int,
    hook_point: str,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int], int]:
    """Load PII dataset, tokenize, align token labels, extract LLM activations.

    Returns:
        all_acts_BLD: [B, L, D] LLM residual stream activations
        token_labels_BL: [B, L] per-token label IDs (-1 for ignored tokens)
        label2id: label name -> ID mapping
        num_classes: number of classes (including O if included)
    """
    print(f"Loading dataset '{config.dataset_name}' split='{config.dataset_split}'...")
    raw = load_dataset(config.dataset_name, split=config.dataset_split)
    assert isinstance(raw, Dataset), "Expected a flat Dataset, not DatasetDict"
    # Filter English only (PII dataset has multi-language data)
    if "language" in raw.column_names:
        raw = raw.filter(lambda x: x["language"] == "English")
        print(f"[DEBUG] After English filter: {len(raw)} samples")
    raw = raw.shuffle(seed=config.random_seed)
    actual_samples = min(config.num_samples, len(raw))
    if actual_samples < config.num_samples:
        print(f"[WARN] Only {actual_samples} samples available (requested {config.num_samples})")
    dataset = raw.select(range(actual_samples))

    texts: list[str] = dataset["source_text"]  # type: ignore[assignment]
    spans_list: list[list[dict]] = dataset["privacy_mask"]  # type: ignore[assignment]

    label2id = build_label2id(include_non_entity=config.include_non_entity)
    num_classes = len(label2id)
    id2label = {v: k for k, v in label2id.items()}
    print(f"[DEBUG] Label mapping ({num_classes} classes): {label2id}")

    # Tokenize with offset mapping for span alignment
    assert model.tokenizer is not None, "Model must have a tokenizer"
    tok_output = model.tokenizer(
        texts,
        max_length=config.context_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    tokens: torch.Tensor = tok_output["input_ids"].to(device)  # type: ignore[union-attr]
    offsets: torch.Tensor = tok_output["offset_mapping"]  # type: ignore[union-attr]

    print(f"[DEBUG] Tokens shape: {tokens.shape}")

    # Align span labels to model tokens
    B, L = tokens.shape
    token_labels_BL = torch.full((B, L), IGNORE_LABEL_ID, dtype=torch.long)
    for b in range(B):
        labels = align_spans_to_tokens(
            offsets[b].tolist(), spans_list[b], label2id, config.include_non_entity  # type: ignore[arg-type]
        )
        token_labels_BL[b] = torch.tensor(labels, dtype=torch.long)

    # Print label distribution stats
    valid_mask = token_labels_BL >= 0
    valid_labels = token_labels_BL[valid_mask]
    total_valid = valid_labels.numel()
    label_counts = torch.bincount(valid_labels, minlength=num_classes)
    print(f"[DEBUG] Token label distribution ({total_valid} valid tokens):")
    for lid in range(num_classes):
        lname = id2label[lid]
        cnt = label_counts[lid].item()
        print(f"[DEBUG]   {lname}: {cnt} ({cnt/total_valid*100:.2f}%)")

    # LLM forward pass
    print(f"[DEBUG] Running LLM forward pass (layer={layer}, hook={hook_point})...")
    acts_list = []
    for i in tqdm(range(0, len(tokens), config.llm_batch_size), desc="LLM Forward"):
        batch_tokens = tokens[i : i + config.llm_batch_size]
        _, cache = model.run_with_cache(
            batch_tokens, names_filter=[hook_point], stop_at_layer=layer + 1
        )
        acts_list.append(cache[hook_point].cpu())
        del cache

    all_acts_BLD = torch.cat(acts_list, dim=0)
    print(f"[DEBUG] LLM activations: shape={all_acts_BLD.shape}, dtype={all_acts_BLD.dtype}")

    return all_acts_BLD, token_labels_BL, label2id, num_classes


# ── Streaming SAE encoding & metric computation ───────────────────────────────

@torch.no_grad()
def encode_and_accumulate(
    sae: SAE,
    all_acts_BLD: torch.Tensor,
    token_labels_BL: torch.Tensor,
    num_classes: int,
    sae_batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Encode SAE activations and accumulate class-level statistics in streaming fashion.

    Instead of materializing [B*L, F] token activations (potentially ~80GB),
    we maintain running sums per class and density counters.

    Returns:
        class_acts: [C, F] sum of feature activations per class
        class_token_counts: [C] number of valid tokens per class
        nonzero_count: [F] number of valid tokens where each feature fires
        total_valid: total number of valid (non-ignored) tokens
    """
    # Get F from a probe encode
    probe = all_acts_BLD[:1, :1, :].to(device)
    F = sae.encode(probe).shape[-1]
    del probe
    print(f"[DEBUG] SAE feature dim F={F}")

    class_acts = np.zeros((num_classes, F), dtype=np.float64)
    class_token_counts = np.zeros(num_classes, dtype=np.int64)
    nonzero_count = np.zeros(F, dtype=np.int64)
    total_valid = 0

    B = all_acts_BLD.shape[0]
    for i in tqdm(range(0, B, sae_batch_size), desc="SAE Encode (streaming)"):
        batch_acts = all_acts_BLD[i : i + sae_batch_size].to(device)
        batch_sae = sae.encode(batch_acts)  # [b, L, F]
        batch_labels = token_labels_BL[i : i + sae_batch_size]  # [b, L]

        if i == 0:
            nonzero_ratio = (batch_sae > 0).float().mean().item()
            print(f"[DEBUG] First batch SAE output: shape={batch_sae.shape}, nonzero_ratio={nonzero_ratio:.6f}")

        # Flatten batch to [b*L, F] and [b*L]
        flat_sae = batch_sae.reshape(-1, F).float().cpu()  # move to CPU for numpy ops
        flat_labels = batch_labels.reshape(-1)  # already on CPU

        del batch_acts, batch_sae

        # Select valid tokens (label >= 0)
        valid_mask = flat_labels >= 0
        if not valid_mask.any():
            continue

        v_acts = flat_sae[valid_mask].numpy()  # [V, F]
        v_labels = flat_labels[valid_mask].numpy()  # [V]
        n_valid = v_acts.shape[0]

        # Accumulate class activations
        for c in range(num_classes):
            cmask = v_labels == c
            n_c = cmask.sum()
            if n_c > 0:
                class_acts[c] += v_acts[cmask].sum(axis=0)
                class_token_counts[c] += n_c

        # Accumulate density (nonzero count)
        nonzero_count += (v_acts > 0).sum(axis=0).astype(np.int64)
        total_valid += n_valid

        del flat_sae, flat_labels, v_acts, v_labels

    print(f"[DEBUG] Streaming accumulation done: total_valid={total_valid}")
    print(f"[DEBUG]   class_token_counts: {class_token_counts.tolist()}")
    print(f"[DEBUG]   nonzero features: {(nonzero_count > 0).sum()}/{F}")

    return class_acts, class_token_counts, nonzero_count, total_valid


def evaluate_from_class_acts(
    class_acts: np.ndarray,
    class_token_counts: np.ndarray,
    nonzero_count: np.ndarray,
    total_valid: int,
    num_classes: int,
    min_feature_density: float,
    max_feature_density: float,
) -> tuple[dict[str, float], list[InfoTheoryResultDetail]]:
    """Compute information theory metrics from pre-aggregated class activations.

    This is equivalent to evaluate_features_information_theory in main.py,
    but works with streaming-accumulated class_acts instead of per-sample activations.

    Args:
        class_acts: [C, F] sum of feature activations per class
        class_token_counts: [C] number of tokens per class (for computing prior Q)
        nonzero_count: [F] number of tokens where each feature fires (for density)
        total_valid: total number of valid tokens
        num_classes: number of classes C
        min_feature_density: lower density threshold
        max_feature_density: upper density threshold
    """
    F = class_acts.shape[1]
    log2_C = np.log2(num_classes)

    # Prior Q: class frequency in the data (token counts)
    Q = class_token_counts.astype(np.float64) / class_token_counts.sum()
    Q = np.clip(Q, 1e-10, 1.0)

    print(f"[DEBUG] evaluate_from_class_acts:")
    print(f"[DEBUG]   F={F}, num_classes={num_classes}, total_valid={total_valid}")
    print(f"[DEBUG]   Prior Q: {Q.tolist()}")

    # Token-level feature density
    token_density_F = nonzero_count.astype(np.float64) / max(total_valid, 1)

    # Alive features: those with non-negligible total activation
    total_activation = class_acts.sum(axis=0)  # [F]
    alive_mask = total_activation > 1e-5
    num_alive = int(alive_mask.sum())

    print(f"[DEBUG]   Alive features: {num_alive}/{F} ({num_alive/F*100:.1f}%)")

    h_norm_all = np.full(F, -1.0)
    kl_all = np.full(F, -1.0)

    if alive_mask.any():
        P = class_acts[:, alive_mask] / total_activation[alive_mask]  # [C, alive_F]
        P = np.clip(P, 1e-10, 1.0)

        h_alive = -np.sum(P * np.log2(P), axis=0) / log2_C
        kl_alive = np.sum(P * np.log2(P / Q[:, None]), axis=0)

        h_norm_all[alive_mask] = h_alive
        kl_all[alive_mask] = kl_alive

        print(f"[DEBUG]   H_norm: [{h_alive.min():.4f}, {h_alive.max():.4f}], mean={h_alive.mean():.4f}")
        print(f"[DEBUG]   KL: [{kl_alive.min():.4f}, {kl_alive.max():.4f}], mean={kl_alive.mean():.4f}")

    # Per-feature details
    feature_details = [
        InfoTheoryResultDetail(
            feature_index=j,
            density=float(token_density_F[j]),
            normalized_entropy=float(h_norm_all[j]),
            kl_divergence=float(kl_all[j]),
        )
        for j in range(F)
    ]

    # Density band-pass filter for aggregate metrics
    density_mask = token_density_F <= max_feature_density
    filtered_mask = alive_mask & density_mask
    num_filtered = int(filtered_mask.sum())

    print(f"[DEBUG]   Density filter [{min_feature_density}, {max_feature_density}]: "
          f"{int(density_mask.sum())} features in band")
    print(f"[DEBUG]   Final filtered (alive & in-band): {num_filtered}/{F} ({num_filtered/F*100:.1f}%)")

    mean_metrics = {
        "mean_kl_divergence": float(np.mean(kl_all[filtered_mask])) if num_filtered else 0.0,
        "mean_normalized_entropy": float(np.mean(h_norm_all[filtered_mask])) if num_filtered else 0.0,
        "alive_features_ratio": num_alive / F,
        "filtered_features_ratio": num_filtered / F,
    }
    print(f"[DEBUG]   mean_metrics: {mean_metrics}")

    return mean_metrics, feature_details


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _get_token_acts_cache_path(
    artifacts_folder: str,
    num_samples: int,
    context_length: int,
    layer: int,
    hook_name: str,
    include_non_entity: bool,
) -> str:
    safe_hook = hook_name.replace(".", "_")
    ne_tag = "withO" if include_non_entity else "noO"
    return os.path.join(
        artifacts_folder,
        f"token_acts_n{num_samples}_ctx{context_length}_layer{layer}_{safe_hook}_{ne_tag}.pt",
    )


# ── Main evaluation loop ──────────────────────────────────────────────────────

def _get_dataset_short_name(dataset_name: str) -> str:
    return dataset_name.split("/")[-1]


def run_eval(
    config: InfoTheoryEvalConfig,
    selected_saes: list[tuple[str, SAE]] | list[tuple[str, str]],
    device: str,
    output_path: str,
    artifacts_path: str,
    force_rerun: bool = False,
):
    eval_instance_id = get_eval_uuid()
    sae_lens_version = get_sae_lens_version()
    sae_bench_commit_hash = get_sae_bench_version()

    print(f"[TOKEN-LEVEL] config: model={config.model_name}, dataset={config.dataset_name}, "
          f"split={config.dataset_split}, include_non_entity={config.include_non_entity}")
    print(f"[TOKEN-LEVEL] num_samples={config.num_samples}, context_length={config.context_length}")
    print(f"[TOKEN-LEVEL] density filter: [{config.min_feature_density}, {config.max_feature_density}]")
    print(f"[TOKEN-LEVEL] Total SAEs to evaluate: {len(selected_saes)}")

    ds_short = _get_dataset_short_name(config.dataset_name)
    ne_tag = "withO" if config.include_non_entity else "noO"
    ds_split_tag = f"{ds_short}_{config.dataset_split}_n{config.num_samples}_ctx{config.context_length}_{ne_tag}"
    output_path = os.path.join(output_path, config.model_name, ds_split_tag)
    os.makedirs(output_path, exist_ok=True)
    artifacts_path = os.path.join(artifacts_path, config.model_name, ds_split_tag)
    os.makedirs(artifacts_path, exist_ok=True)

    llm_dtype = general_utils.str_to_dtype(config.llm_dtype)
    print(f"[TOKEN-LEVEL] Loading model {config.model_name} with dtype={llm_dtype}...")
    model = HookedTransformer.from_pretrained_no_processing(
        config.model_name, device=device, dtype=llm_dtype
    )

    results_dict = {}

    # Cache: same layer activations + token labels computed once
    cached_layer: int | None = None
    cached_hook: str | None = None
    all_acts_BLD: torch.Tensor | None = None
    token_labels_BL: torch.Tensor | None = None
    label2id: dict[str, int] | None = None
    num_classes: int = 0

    for sae_release, sae_object_or_id in tqdm(selected_saes, desc="Running SAE evaluation"):
        sae_id, sae, _ = general_utils.load_and_format_sae(  # type: ignore[misc]
            sae_release, sae_object_or_id, device
        )
        sae = sae.to(device=device, dtype=llm_dtype)

        layer = sae.cfg.hook_layer
        hook_name = sae.cfg.hook_name

        print(f"\n[TOKEN-LEVEL] === Evaluating SAE: {sae_release}_{sae_id} ===")
        print(f"[TOKEN-LEVEL] layer={layer}, hook={hook_name}, d_sae={sae.cfg.d_sae}")

        sae_result_path = general_utils.get_results_filepath(output_path, sae_release, sae_id)
        if os.path.exists(sae_result_path) and not force_rerun:
            print(f"Skipping {sae_release}_{sae_id} (results exist)")
            continue

        if layer != cached_layer or hook_name != cached_hook:
            cache_path = _get_token_acts_cache_path(
                artifacts_path, config.num_samples, config.context_length,
                layer, hook_name, config.include_non_entity,
            )
            print(f"[TOKEN-LEVEL] Activation cache: {cache_path}")

            if os.path.exists(cache_path) and not force_rerun:
                print(f"Loading cached token-level activations from {cache_path}")
                cache_data = torch.load(cache_path, map_location="cpu", weights_only=False)
                all_acts_BLD = cache_data["acts"]
                token_labels_BL = cache_data["token_labels"]
                label2id = cache_data["label2id"]
                num_classes = int(cache_data["num_classes"])
                print(f"[TOKEN-LEVEL] Loaded: acts={all_acts_BLD.shape}, num_classes={num_classes}")  # type: ignore[union-attr]
            else:
                print(f"Computing token-level activations for layer {layer}...")
                all_acts_BLD, token_labels_BL, label2id, num_classes = get_token_level_activations(
                    model, config, layer, hook_name, device
                )
                torch.save(
                    {
                        "acts": all_acts_BLD,
                        "token_labels": token_labels_BL,
                        "label2id": label2id,
                        "num_classes": num_classes,
                    },
                    cache_path,
                )
                print(f"[TOKEN-LEVEL] Saved cache to {cache_path} "
                      f"({os.path.getsize(cache_path)/1024/1024:.1f}MB)")

            cached_layer = layer
            cached_hook = hook_name

        sae_cfg_dict = sae.cfg.to_dict()

        assert all_acts_BLD is not None and token_labels_BL is not None
        class_acts, class_token_counts, nonzero_count, total_valid = encode_and_accumulate(
            sae, all_acts_BLD, token_labels_BL,
            num_classes, config.sae_batch_size, device,
        )
        del sae
        torch.cuda.empty_cache()

        mean_metrics, feature_details = evaluate_from_class_acts(
            class_acts, class_token_counts, nonzero_count, total_valid,
            num_classes, config.min_feature_density, config.max_feature_density,
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
        print(f"[TOKEN-LEVEL] Results saved to {sae_result_path}")

        gc.collect()
        torch.cuda.empty_cache()

    return results_dict


# ── CLI ────────────────────────────────────────────────────────────────────────

def create_config_and_selected_saes(args) -> tuple[InfoTheoryEvalConfig, list[tuple[str, str]]]:
    config = InfoTheoryEvalConfig(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        text_column="source_text",
        label_column="privacy_mask",
        num_samples=args.num_samples,
        min_feature_density=args.min_feature_density,
        max_feature_density=args.max_feature_density,
        label_type="token",
        include_non_entity=args.include_non_entity,
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

    print(f"[TOKEN-LEVEL] Config: {config}")
    print(f"[TOKEN-LEVEL] Selected {len(selected_saes)} SAE(s)")

    return config, selected_saes


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Token-level information theory evaluation for SAE features (PII/NER datasets)"
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--sae_regex_pattern", type=str, required=True)
    parser.add_argument("--sae_block_pattern", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="ai4privacy/pii-masking-300k",
                        help="HuggingFace dataset with token-level labels")
    parser.add_argument("--dataset_split", type=str, default="validation")
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--include_non_entity", action="store_true", default=True,
                        help="Include 'O' (non-entity) tokens as a class (default: True)")
    parser.add_argument("--exclude_non_entity", dest="include_non_entity", action="store_false",
                        help="Exclude 'O' tokens, only evaluate on entity tokens")
    parser.add_argument("--min_feature_density", type=float, default=1e-4)
    parser.add_argument("--max_feature_density", type=float, default=1e-2)
    parser.add_argument("--output_folder", type=str, default="eval_results/info_theory")
    parser.add_argument("--artifacts_path", type=str, default="artifacts/info_theory")
    parser.add_argument("--llm_batch_size", type=int, default=None)
    parser.add_argument("--llm_dtype", type=str, default=None)
    parser.add_argument("--force_rerun", action="store_true")
    return parser


if __name__ == "__main__":
    args = arg_parser().parse_args()
    device = general_utils.setup_environment()
    print(f"[TOKEN-LEVEL] Device: {device}")
    config, selected_saes = create_config_and_selected_saes(args)
    run_eval(config, selected_saes, device, args.output_folder, args.artifacts_path, args.force_rerun)
