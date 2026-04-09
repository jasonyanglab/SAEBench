"""Top-k Feature Precision/Recall Verification for H/KL Metrics.

This script validates that features ranked highly by KL divergence truly encode
their assigned concepts, by computing token-level Precision and Recall.

Experiment design:
    1. Load existing H/KL results to get per-feature KL, density, P(c|j)
    2. Build candidate pool: alive features with density <= max_density (filtered_mask)
    3. Assign each feature to its primary class: c_j = argmax_c P(c|j)
    4. For each class, build FOUR ranked lists and pick top-k from each:
        - KL-top         : raw KL descending (original)
        - KL+floor-top   : KL descending, restricted to density >= min_density
        - MI-top         : (density × KL) descending — support-weighted KL
        - Density-top    : density descending (frequency confound baseline)
       Plus Random baseline for reference.
    5. Stream SAE encode, compute frequency-based Precision and Recall
    6. Compare all groups (Recall is the primary, non-circular validation metric)

Key metrics (all frequency-based, NOT amplitude-based):
    - Precision_j = TP / (TP + FP)  where TP/FP count tokens, not activation magnitudes
    - Recall_j    = TP / (TP + FN)  — this is the PRIMARY validation metric
    - Joint Recall@k: OR-union of top-k features' activations

Why Recall is the core metric:
    - P(c|j) answers Feature→Concept (used to compute KL/H)
    - Recall answers Concept→Feature (independent, orthogonal direction)
    - No circular reasoning: high KL does NOT mathematically imply high Recall

Why Precision uses frequency (not amplitude):
    - Amplitude Precision = P(c|j) itself → circular with KL
    - Frequency Precision counts activation events → partially independent
"""

import argparse
import json
import gc
import os
import time
from collections import defaultdict

import numpy as np
import torch
from sae_lens import SAE
from tqdm import tqdm
from transformer_lens import HookedTransformer

import sae_bench.sae_bench_utils.activation_collection as activation_collection
import sae_bench.sae_bench_utils.general_utils as general_utils
from sae_bench.sae_bench_utils.sae_selection_utils import get_saes_from_regex

from sae_bench.evals.info_theory.eval_config import InfoTheoryEvalConfig
from sae_bench.evals.info_theory.main_token import (
    build_label2id,
    get_token_level_activations,
    encode_and_accumulate,
    _get_token_acts_cache_path,
    PII_ENTITY_TYPES,
)


# ── Step 1: Load H/KL results and build top-k feature lists ─────────────────

def load_hkl_results(result_path: str) -> dict:
    """Load a single SAE's H/KL evaluation result JSON."""
    with open(result_path, "r") as f:
        return json.load(f)


def build_topk_candidates(
    hkl_result: dict,
    max_feature_density: float,
) -> dict:
    """From H/KL results, build candidate feature pool.

    Candidate pool = alive (KL >= 0) AND density <= max_feature_density,
    matching the filtered_mask used in mean_KL computation.

    Note: class assignment and top-k selection are deferred to
    select_topk_from_class_acts(), because we need class_acts (from a
    SAE encode pass) to compute argmax P(c|j).

    Args:
        hkl_result: loaded H/KL eval result JSON
        max_feature_density: upper density threshold (same as H/KL eval config)

    Returns:
        dict with keys:
            'num_classes': int
            'candidates': set of feature indices in candidate pool
            'feature_kl': {feature_idx: KL_value}
            'feature_density': {feature_idx: density_value}
            'F': total number of features
    """
    details = hkl_result["eval_result_details"]

    # Reconstruct per-feature KL, density
    feature_kl = {}
    feature_density = {}
    for d in details:
        idx = d["feature_index"]
        kl = d["kl_divergence"]
        density = d["density"]
        feature_kl[idx] = kl
        feature_density[idx] = density

    F = len(details)

    # Determine number of classes from eval config
    include_non_entity = hkl_result["eval_config"].get("include_non_entity", False)
    if include_non_entity:
        num_classes = len(PII_ENTITY_TYPES) + 1  # +1 for "O"
    else:
        num_classes = len(PII_ENTITY_TYPES)  # 25

    # Build candidate pool: alive AND density <= threshold
    candidates = set()
    for idx in range(F):
        kl = feature_kl.get(idx, -1.0)
        density = feature_density.get(idx, 0.0)
        if kl >= 0 and density <= max_feature_density:
            candidates.add(idx)

    print(f"[TOPK] Candidate pool: {len(candidates)}/{F} features "
          f"(alive & density <= {max_feature_density})")

    return {
        "num_classes": num_classes,
        "candidates": candidates,
        "feature_kl": feature_kl,
        "feature_density": feature_density,
        "F": F,
    }


def select_topk_from_class_acts(
    class_acts: np.ndarray,
    candidates: set,
    feature_kl: dict,
    feature_density: dict,
    k_values: list[int],
    num_classes: int,
    id2label: dict[int, str],
    min_density: float = 0.0,
) -> dict:
    """Assign features to classes via argmax P(c|j), then select top-k per class.

    Produces FOUR ranked lists per class (all sharing the same candidate pool
    used for class assignment):
        - KL-top: sorted by KL descending (raw KL — original experiment group)
        - Density-top: sorted by density descending (frequency confound baseline)
        - KL+floor-top: sorted by KL descending, but only features with
            density >= min_density (support-floored KL)
        - MI-top: sorted by (density * KL) descending (mutual-information style,
            equivalent to support-weighted KL)

    Args:
        class_acts: [C, F] sum of feature activations per class
        candidates: set of feature indices in candidate pool
        feature_kl: {feature_idx: KL_value}
        feature_density: {feature_idx: density_value}
        k_values: e.g. [1, 5, 10, 20]
        num_classes: C
        id2label: {class_id: label_name}
        min_density: support floor for KL+floor variant. Features with
            density < min_density are excluded from the KL+floor ranking only;
            other rankings ignore this threshold.

    Returns:
        dict with:
            'feature_class': {feat_idx: class_id} for all candidates
            'topk_per_class': {class_id: {k: [feat_indices]}} (KL-ranked)
            'density_topk_per_class': {class_id: {k: [feat_indices]}}
            'kl_floor_topk_per_class': {class_id: {k: [feat_indices]}}
            'mi_topk_per_class': {class_id: {k: [feat_indices]}}
            'class_candidate_counts': {class_id: num_candidates}
            'class_floor_counts': {class_id: num_candidates with density>=min_density}
    """
    # Compute P(c|j) for candidates
    total_act = class_acts.sum(axis=0)  # [F]

    feature_class = {}
    class_candidates_kl = defaultdict(list)        # class_id -> [(kl, feat_idx)]
    class_candidates_density = defaultdict(list)   # class_id -> [(density, feat_idx)]
    class_candidates_kl_floor = defaultdict(list)  # class_id -> [(kl, feat_idx)] subject to density>=min_density
    class_candidates_mi = defaultdict(list)        # class_id -> [(density*kl, feat_idx)]

    for j in candidates:
        if total_act[j] < 1e-10:
            continue
        p_cj = class_acts[:, j] / total_act[j]  # [C]
        c_j = int(np.argmax(p_cj))
        feature_class[j] = c_j
        kl_j = feature_kl[j]
        d_j = feature_density.get(j, 0.0)
        class_candidates_kl[c_j].append((kl_j, j))
        class_candidates_density[c_j].append((d_j, j))
        class_candidates_mi[c_j].append((d_j * kl_j, j))
        if d_j >= min_density:
            class_candidates_kl_floor[c_j].append((kl_j, j))

    # Sort all four lists descending by their primary score
    for c in class_candidates_kl:
        class_candidates_kl[c].sort(key=lambda x: x[0], reverse=True)
        class_candidates_density[c].sort(key=lambda x: x[0], reverse=True)
        class_candidates_mi[c].sort(key=lambda x: x[0], reverse=True)
    for c in class_candidates_kl_floor:
        class_candidates_kl_floor[c].sort(key=lambda x: x[0], reverse=True)

    topk_per_class = {}
    density_topk_per_class = {}
    kl_floor_topk_per_class = {}
    mi_topk_per_class = {}
    class_candidate_counts = {}
    class_floor_counts = {}

    print(f"\n[TOPK] Feature-to-class assignment & top-k selection (min_density={min_density}):")
    for c in range(num_classes):
        cands_kl = class_candidates_kl.get(c, [])
        cands_den = class_candidates_density.get(c, [])
        cands_floor = class_candidates_kl_floor.get(c, [])
        cands_mi = class_candidates_mi.get(c, [])
        class_candidate_counts[c] = len(cands_kl)
        class_floor_counts[c] = len(cands_floor)
        topk_per_class[c] = {}
        density_topk_per_class[c] = {}
        kl_floor_topk_per_class[c] = {}
        mi_topk_per_class[c] = {}
        for k in k_values:
            topk_per_class[c][k] = [idx for _, idx in cands_kl[:k]]
            density_topk_per_class[c][k] = [idx for _, idx in cands_den[:k]]
            kl_floor_topk_per_class[c][k] = [idx for _, idx in cands_floor[:k]]
            mi_topk_per_class[c][k] = [idx for _, idx in cands_mi[:k]]

        label = id2label.get(c, f"class_{c}")
        top1_kl = f"{cands_kl[0][0]:.4f}" if cands_kl else "N/A"
        top1_floor = f"{cands_floor[0][0]:.4f}" if cands_floor else "N/A"
        top1_mi = f"{cands_mi[0][0]:.4f}" if cands_mi else "N/A"
        print(f"[TOPK]   {label:15s}: {len(cands_kl):5d} cands ({len(cands_floor):4d} w/ floor), "
              f"top-1 KL={top1_kl}, KL+floor={top1_floor}, MI={top1_mi}")

    return {
        "feature_class": feature_class,
        "topk_per_class": topk_per_class,
        "density_topk_per_class": density_topk_per_class,
        "kl_floor_topk_per_class": kl_floor_topk_per_class,
        "mi_topk_per_class": mi_topk_per_class,
        "class_candidate_counts": class_candidate_counts,
        "class_floor_counts": class_floor_counts,
    }


# ── Step 2: Streaming P/R computation ────────────────────────────────────────

@torch.no_grad()
def compute_precision_recall_streaming(
    sae: SAE,
    all_acts_BLD: torch.Tensor,
    token_labels_BL: torch.Tensor,
    feature_class: dict[int, int],
    topk_per_class: dict[int, dict[int, list[int]]],
    density_topk_per_class: dict[int, dict[int, list[int]]],
    kl_floor_topk_per_class: dict[int, dict[int, list[int]]],
    mi_topk_per_class: dict[int, dict[int, list[int]]],
    class_candidate_counts: dict[int, int],
    class_floor_counts: dict[int, int],
    feature_kl: dict[int, float],
    num_classes: int,
    k_values: list[int],
    sae_batch_size: int,
    device: str,
    n_random_trials: int = 10,
    random_seed: int = 42,
) -> dict:
    """Stream SAE encode and compute frequency-based Precision/Recall for top-k features.

    Single-pass design: simultaneously computes
        - Per-feature TP/FP/FN (for single-feature P/R at k=1)
        - Joint OR-union TP/FP for KL-top, Density-top, and Random baselines

    All P/R are frequency-based (count activation events), NOT amplitude-based,
    to avoid circular reasoning with KL (which is derived from amplitude P(c|j)).

    Args:
        sae: loaded SAE model, used for encode() to get feature activations
        all_acts_BLD: [B, L, D] LLM residual stream activations (CPU tensor).
            B=num_samples, L=context_length, D=model hidden dim
        token_labels_BL: [B, L] per-token class label IDs (CPU tensor).
            -1 = ignored token (O tokens in noO mode), 0..C-1 = valid class IDs
        feature_class: {feature_idx: class_id} mapping each candidate feature
            to its primary class via argmax P(c|j). Only includes features in
            the candidate pool (alive & low density)
        topk_per_class: {class_id: {k: [feature_indices]}} KL-ranked top-k
            feature lists per class. E.g. topk_per_class[3][5] = top-5 features
            for class 3, sorted by KL descending (experiment group)
        density_topk_per_class: {class_id: {k: [feature_indices]}} same structure
            as topk_per_class, but ranked by density descending (frequency
            confound control baseline)
        kl_floor_topk_per_class: {class_id: {k: [feature_indices]}} KL-ranked,
            but only over features with density >= min_density (support floor)
        mi_topk_per_class: {class_id: {k: [feature_indices]}} ranked by
            (density × KL), i.e. mutual-information-style scoring
        class_candidate_counts: {class_id: num_candidates} total number of
            candidate features assigned to each class (for reporting)
        class_floor_counts: {class_id: num_candidates with density>=min_density}
        feature_kl: {feature_idx: KL_value} per-feature KL divergence from
            H/KL eval results (used only for metadata in single-feature output)
        num_classes: C, total number of classes (25 in noO PII mode)
        k_values: list of k values to evaluate, e.g. [1, 5, 10, 20]
        sae_batch_size: number of samples per SAE encode batch
        device: 'cuda' or 'cpu' for SAE encode
        n_random_trials: number of independent random sampling trials for
            the Random baseline (default 10, results averaged)
        random_seed: seed for reproducible random feature sampling

    Returns:
        dict with structure:
            'k_values': list[int] — echo of input k_values
            'n_random_trials': int — echo of input
            'class_token_counts': {class_id: int} — number of valid tokens per class
            'per_class': {class_id: {
                'n_candidates': int — total candidates assigned to this class
                'single_feature': {  — P/R for the single KL-top-1 feature
                    'feature_idx': int,
                    'kl': float,
                    'precision': float,  — TP / (TP + FP), frequency-based
                    'recall': float,     — TP / (TP + FN), frequency-based
                }
                'joint': {k: {  — Joint OR-union P/R for top-k features
                    'n_features_used': int,
                    'n_features_used_floor': int,
                    'n_features_used_mi': int,
                    'kl_top_precision': float,
                    'kl_top_recall': float,
                    'density_top_precision': float,
                    'density_top_recall': float,
                    'kl_floor_top_precision': float,
                    'kl_floor_top_recall': float,
                    'mi_top_precision': float,
                    'mi_top_recall': float,
                    'random_precision_mean': float,
                    'random_precision_std': float,
                    'random_recall_mean': float,
                    'random_recall_std': float,
                }}
            }}
            'macro': {k: {  — Macro-averaged across classes (equal weight per class)
                'n_classes_evaluated': int,
                'n_classes_evaluated_floor': int,
                'kl_top_precision': float,
                'kl_top_recall': float,
                'density_top_precision': float,
                'density_top_recall': float,
                'kl_floor_top_precision': float,
                'kl_floor_top_recall': float,
                'mi_top_precision': float,
                'mi_top_recall': float,
                'random_precision': float,
                'random_recall': float,
                'recall_delta_kl_vs_random': float,
                'recall_delta_kl_vs_density': float,
                'recall_delta_floor_vs_kl': float,
                'recall_delta_mi_vs_kl': float,
            }}
    """
    B = all_acts_BLD.shape[0]

    # Collect all candidate features assigned to each class (for random baseline)
    class_all_features = defaultdict(list)
    for feat_idx, c in feature_class.items():
        class_all_features[c].append(feat_idx)

    # Build the set of all feature indices we need to track
    # (KL-top + Density-top + KL+floor-top + MI-top + all candidates for random sampling)
    tracked_features = set()
    for c in range(num_classes):
        for k in k_values:
            tracked_features.update(topk_per_class.get(c, {}).get(k, []))
            tracked_features.update(density_topk_per_class.get(c, {}).get(k, []))
            tracked_features.update(kl_floor_topk_per_class.get(c, {}).get(k, []))
            tracked_features.update(mi_topk_per_class.get(c, {}).get(k, []))
        tracked_features.update(class_all_features.get(c, []))
    tracked_features = sorted(tracked_features)
    feat_to_local = {f: i for i, f in enumerate(tracked_features)}
    n_tracked = len(tracked_features)

    print(f"\n[P/R] Tracking {n_tracked} features for P/R computation")
    print(f"[P/R] Streaming SAE encode over {B} samples, batch_size={sae_batch_size}")

    # ── Pre-compute local index arrays for vectorized lookup ──────────────

    # Per-feature counters (for single-feature P/R)
    feat_class_act_count = np.zeros((n_tracked, num_classes), dtype=np.int64)
    feat_act_count = np.zeros(n_tracked, dtype=np.int64)
    class_token_count = np.zeros(num_classes, dtype=np.int64)

    # Joint OR-union counters: KL-top
    joint_tp = {c: {k: 0 for k in k_values} for c in range(num_classes)}
    joint_fp = {c: {k: 0 for k in k_values} for c in range(num_classes)}

    # Joint OR-union counters: Density-top
    density_tp = {c: {k: 0 for k in k_values} for c in range(num_classes)}
    density_fp = {c: {k: 0 for k in k_values} for c in range(num_classes)}

    # Joint OR-union counters: KL+floor-top
    kl_floor_tp = {c: {k: 0 for k in k_values} for c in range(num_classes)}
    kl_floor_fp = {c: {k: 0 for k in k_values} for c in range(num_classes)}

    # Joint OR-union counters: MI-top
    mi_tp = {c: {k: 0 for k in k_values} for c in range(num_classes)}
    mi_fp = {c: {k: 0 for k in k_values} for c in range(num_classes)}

    # Pre-compute local index arrays for all four ranking sets
    topk_local = {}            # (c, k) -> np.array of local indices
    density_topk_local = {}    # (c, k) -> np.array of local indices
    kl_floor_topk_local = {}   # (c, k) -> np.array of local indices
    mi_topk_local = {}         # (c, k) -> np.array of local indices
    for c in range(num_classes):
        for k in k_values:
            feats = topk_per_class.get(c, {}).get(k, [])
            topk_local[(c, k)] = np.array([feat_to_local[f] for f in feats]) if feats else None

            den_feats = density_topk_per_class.get(c, {}).get(k, [])
            density_topk_local[(c, k)] = np.array([feat_to_local[f] for f in den_feats]) if den_feats else None

            floor_feats = kl_floor_topk_per_class.get(c, {}).get(k, [])
            kl_floor_topk_local[(c, k)] = np.array([feat_to_local[f] for f in floor_feats]) if floor_feats else None

            mi_feats = mi_topk_per_class.get(c, {}).get(k, [])
            mi_topk_local[(c, k)] = np.array([feat_to_local[f] for f in mi_feats]) if mi_feats else None

    # Pre-generate random feature sets and their local indices
    rng = np.random.RandomState(random_seed)
    random_local = {}  # (c, k, trial) -> np.array of local indices or None
    for c in range(num_classes):
        all_feats_c = class_all_features.get(c, [])
        for k in k_values:
            for trial in range(n_random_trials):
                n_sample = min(k, len(all_feats_c))
                if n_sample > 0:
                    sampled = rng.choice(all_feats_c, size=n_sample, replace=False)
                    random_local[(c, k, trial)] = np.array([feat_to_local[f] for f in sampled])
                else:
                    random_local[(c, k, trial)] = None

    random_tp = {c: {k: np.zeros(n_random_trials, dtype=np.int64) for k in k_values}
                 for c in range(num_classes)}
    random_fp = {c: {k: np.zeros(n_random_trials, dtype=np.int64) for k in k_values}
                 for c in range(num_classes)}

    tracked_indices_tensor = torch.tensor(tracked_features, dtype=torch.long, device=device)

    # ── Single streaming pass ─────────────────────────────────────────────

    for i in tqdm(range(0, B, sae_batch_size), desc="SAE Encode (P/R)"):
        batch_acts = all_acts_BLD[i : i + sae_batch_size].to(device)
        batch_sae = sae.encode(batch_acts)  # [b, L, F]
        batch_labels = token_labels_BL[i : i + sae_batch_size]

        # Extract only tracked features: [b, L, n_tracked]
        batch_tracked = batch_sae[:, :, tracked_indices_tensor].cpu()
        flat_tracked = batch_tracked.reshape(-1, n_tracked)
        flat_labels = batch_labels.reshape(-1)

        del batch_acts, batch_sae, batch_tracked

        valid_mask = flat_labels >= 0
        if not valid_mask.any():
            continue

        v_tracked = flat_tracked[valid_mask].float().numpy()  # [V, n_tracked]
        v_labels = flat_labels[valid_mask].numpy()             # [V]
        v_active = (v_tracked > 0)                     # [V, n_tracked] bool

        del flat_tracked, flat_labels

        # Accumulate per-feature and joint counts for each class
        for c in range(num_classes):
            cmask = (v_labels == c)
            n_c = int(cmask.sum())
            if n_c > 0:
                class_token_count[c] += n_c
                feat_class_act_count[:, c] += v_active[cmask].sum(axis=0).astype(np.int64)

            not_cmask = ~cmask

            # Joint OR-union for each k
            for k in k_values:
                # KL-top
                local_idx = topk_local[(c, k)]
                if local_idx is not None:
                    joint_active = v_active[:, local_idx].any(axis=1)
                    joint_tp[c][k] += int((joint_active & cmask).sum())
                    joint_fp[c][k] += int((joint_active & not_cmask).sum())

                # Density-top
                den_idx = density_topk_local[(c, k)]
                if den_idx is not None:
                    den_active = v_active[:, den_idx].any(axis=1)
                    density_tp[c][k] += int((den_active & cmask).sum())
                    density_fp[c][k] += int((den_active & not_cmask).sum())

                # KL+floor-top
                floor_idx = kl_floor_topk_local[(c, k)]
                if floor_idx is not None:
                    floor_active = v_active[:, floor_idx].any(axis=1)
                    kl_floor_tp[c][k] += int((floor_active & cmask).sum())
                    kl_floor_fp[c][k] += int((floor_active & not_cmask).sum())

                # MI-top
                mi_idx = mi_topk_local[(c, k)]
                if mi_idx is not None:
                    mi_active = v_active[:, mi_idx].any(axis=1)
                    mi_tp[c][k] += int((mi_active & cmask).sum())
                    mi_fp[c][k] += int((mi_active & not_cmask).sum())

                # Random baselines
                for trial in range(n_random_trials):
                    rl = random_local[(c, k, trial)]
                    if rl is None:
                        continue
                    rand_active = v_active[:, rl].any(axis=1)
                    random_tp[c][k][trial] += int((rand_active & cmask).sum())
                    random_fp[c][k][trial] += int((rand_active & not_cmask).sum())

        feat_act_count += v_active.sum(axis=0).astype(np.int64)
        del v_tracked, v_labels, v_active

    print(f"[P/R] Streaming done. Class token counts: {class_token_count.tolist()}")

    # ── Assemble results ─────────────────────────────────────────────────────

    results = {
        "k_values": k_values,
        "n_random_trials": n_random_trials,
        "class_token_counts": {int(c): int(class_token_count[c]) for c in range(num_classes)},
        "per_class": {},
        "macro": {},
    }

    # Single-feature results (k=1 top feature per class)
    single_feature_results = {}
    for c in range(num_classes):
        top1_feats = topk_per_class.get(c, {}).get(1, [])
        if top1_feats:
            li = feat_to_local[top1_feats[0]]
            tp = int(feat_class_act_count[li, c])
            total_act_c = int(feat_act_count[li])
            fp = total_act_c - tp
            fn = int(class_token_count[c]) - tp
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            single_feature_results[c] = {
                "feature_idx": top1_feats[0],
                "kl": feature_kl[top1_feats[0]],
                "precision": p,
                "recall": r,
            }

    # Per-class joint results
    for c in range(num_classes):
        per_k = {}
        for k in k_values:
            n_feats = len(topk_per_class.get(c, {}).get(k, []))
            n_cls_tokens = int(class_token_count[c])

            # KL-top
            tp = joint_tp[c][k]
            fp = joint_fp[c][k]
            fn = n_cls_tokens - tp
            kl_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            kl_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            # Density-top
            d_tp = density_tp[c][k]
            d_fp = density_fp[c][k]
            d_fn = n_cls_tokens - d_tp
            den_p = d_tp / (d_tp + d_fp) if (d_tp + d_fp) > 0 else 0.0
            den_r = d_tp / (d_tp + d_fn) if (d_tp + d_fn) > 0 else 0.0

            # KL+floor-top
            n_floor_feats = len(kl_floor_topk_per_class.get(c, {}).get(k, []))
            f_tp = kl_floor_tp[c][k]
            f_fp = kl_floor_fp[c][k]
            f_fn = n_cls_tokens - f_tp
            floor_p = f_tp / (f_tp + f_fp) if (f_tp + f_fp) > 0 else 0.0
            floor_r = f_tp / (f_tp + f_fn) if (f_tp + f_fn) > 0 else 0.0

            # MI-top
            n_mi_feats = len(mi_topk_per_class.get(c, {}).get(k, []))
            m_tp = mi_tp[c][k]
            m_fp = mi_fp[c][k]
            m_fn = n_cls_tokens - m_tp
            mi_p = m_tp / (m_tp + m_fp) if (m_tp + m_fp) > 0 else 0.0
            mi_r = m_tp / (m_tp + m_fn) if (m_tp + m_fn) > 0 else 0.0

            # Random baseline (average over trials)
            rand_p_list = []
            rand_r_list = []
            for trial in range(n_random_trials):
                r_tp = int(random_tp[c][k][trial])
                r_fp = int(random_fp[c][k][trial])
                r_fn = n_cls_tokens - r_tp
                rp = r_tp / (r_tp + r_fp) if (r_tp + r_fp) > 0 else 0.0
                rr = r_tp / (r_tp + r_fn) if (r_tp + r_fn) > 0 else 0.0
                rand_p_list.append(rp)
                rand_r_list.append(rr)

            per_k[k] = {
                "n_features_used": n_feats,
                "n_features_used_floor": n_floor_feats,
                "n_features_used_mi": n_mi_feats,
                "kl_top_precision": kl_p,
                "kl_top_recall": kl_r,
                "density_top_precision": den_p,
                "density_top_recall": den_r,
                "kl_floor_top_precision": floor_p,
                "kl_floor_top_recall": floor_r,
                "mi_top_precision": mi_p,
                "mi_top_recall": mi_r,
                "random_precision_mean": float(np.mean(rand_p_list)) if rand_p_list else 0.0,
                "random_precision_std": float(np.std(rand_p_list)) if rand_p_list else 0.0,
                "random_recall_mean": float(np.mean(rand_r_list)) if rand_r_list else 0.0,
                "random_recall_std": float(np.std(rand_r_list)) if rand_r_list else 0.0,
            }

        results["per_class"][c] = {
            "n_candidates": class_candidate_counts.get(c, 0),
            "n_candidates_floor": class_floor_counts.get(c, 0),
            "single_feature": single_feature_results.get(c, {}),
            "joint": per_k,
        }

    # Macro averages (over classes with at least 1 candidate)
    # Note: classes are included if they have a KL-top candidate. The floor
    # group may be empty for some classes; for those classes the floor P/R
    # are NaN-ish (0/0 → 0). We average only over classes whose floor group
    # is non-empty for the floor metrics, to avoid biasing it down.
    for k in k_values:
        kl_p_list, kl_r_list = [], []
        den_p_list, den_r_list = [], []
        floor_p_list, floor_r_list = [], []
        mi_p_list, mi_r_list = [], []
        rand_p_list, rand_r_list = [], []
        for c in range(num_classes):
            per_k = results["per_class"].get(c, {}).get("joint", {}).get(k, {})
            if per_k and per_k.get("n_features_used", 0) > 0:
                kl_p_list.append(per_k["kl_top_precision"])
                kl_r_list.append(per_k["kl_top_recall"])
                den_p_list.append(per_k["density_top_precision"])
                den_r_list.append(per_k["density_top_recall"])
                mi_p_list.append(per_k["mi_top_precision"])
                mi_r_list.append(per_k["mi_top_recall"])
                rand_p_list.append(per_k["random_precision_mean"])
                rand_r_list.append(per_k["random_recall_mean"])
            if per_k and per_k.get("n_features_used_floor", 0) > 0:
                floor_p_list.append(per_k["kl_floor_top_precision"])
                floor_r_list.append(per_k["kl_floor_top_recall"])

        results["macro"][k] = {
            "n_classes_evaluated": len(kl_p_list),
            "n_classes_evaluated_floor": len(floor_p_list),
            "kl_top_precision": float(np.mean(kl_p_list)) if kl_p_list else 0.0,
            "kl_top_recall": float(np.mean(kl_r_list)) if kl_r_list else 0.0,
            "density_top_precision": float(np.mean(den_p_list)) if den_p_list else 0.0,
            "density_top_recall": float(np.mean(den_r_list)) if den_r_list else 0.0,
            "kl_floor_top_precision": float(np.mean(floor_p_list)) if floor_p_list else 0.0,
            "kl_floor_top_recall": float(np.mean(floor_r_list)) if floor_r_list else 0.0,
            "mi_top_precision": float(np.mean(mi_p_list)) if mi_p_list else 0.0,
            "mi_top_recall": float(np.mean(mi_r_list)) if mi_r_list else 0.0,
            "random_precision": float(np.mean(rand_p_list)) if rand_p_list else 0.0,
            "random_recall": float(np.mean(rand_r_list)) if rand_r_list else 0.0,
            "recall_delta_kl_vs_random": float(np.mean(kl_r_list) - np.mean(rand_r_list)) if kl_r_list else 0.0,
            "recall_delta_kl_vs_density": float(np.mean(kl_r_list) - np.mean(den_r_list)) if kl_r_list else 0.0,
            "recall_delta_floor_vs_kl": float(np.mean(floor_r_list) - np.mean(kl_r_list)) if floor_r_list and kl_r_list else 0.0,
            "recall_delta_mi_vs_kl": float(np.mean(mi_r_list) - np.mean(kl_r_list)) if mi_r_list and kl_r_list else 0.0,
        }

    return results


# ── Pretty-print results ─────────────────────────────────────────────────────

def print_results_summary(results: dict, id2label: dict[int, str]):
    """Print a human-readable summary of the P/R verification results."""
    print("\n" + "=" * 80)
    print("TOP-K FEATURE PRECISION/RECALL VERIFICATION RESULTS")
    print("=" * 80)

    # Macro summary
    print("\n── Macro-Averaged Results (across classes) ──")
    header = (f"{'k':>3s} | {'KL P':>6s} {'KL R':>6s} | "
              f"{'Flr P':>6s} {'Flr R':>6s} | "
              f"{'MI P':>6s} {'MI R':>6s} | "
              f"{'Den P':>6s} {'Den R':>6s} | "
              f"{'Rnd P':>6s} {'Rnd R':>6s}")
    print(header)
    print("-" * len(header))
    for k in results["k_values"]:
        m = results["macro"].get(str(k), results["macro"].get(k, {}))
        print(f"{k:>3d} | "
              f"{m.get('kl_top_precision', 0):>6.3f} {m.get('kl_top_recall', 0):>6.3f} | "
              f"{m.get('kl_floor_top_precision', 0):>6.3f} {m.get('kl_floor_top_recall', 0):>6.3f} | "
              f"{m.get('mi_top_precision', 0):>6.3f} {m.get('mi_top_recall', 0):>6.3f} | "
              f"{m.get('density_top_precision', 0):>6.3f} {m.get('density_top_recall', 0):>6.3f} | "
              f"{m.get('random_precision', 0):>6.3f} {m.get('random_recall', 0):>6.3f}")

    first_k = results["k_values"][0]
    n_eval = results["macro"].get(str(first_k), results["macro"].get(first_k, {})).get("n_classes_evaluated", 0)
    print(f"\n  ({n_eval} classes evaluated)")

    # Per-class details for k=1 (single feature)
    print("\n── Per-Class Single Feature (k=1) Results ──")
    print(f"{'Class':>15s} | {'#Cands':>7s} {'#Tokens':>8s} | "
          f"{'feat_idx':>8s} {'KL':>8s} | {'Prec':>8s} {'Recall':>8s}")
    print("-" * 80)
    for c_str, data in sorted(results["per_class"].items(), key=lambda x: int(x[0])):
        c = int(c_str)
        label = id2label.get(c, f"class_{c}")
        sf = data.get("single_feature", {})
        n_cands = data.get("n_candidates", 0)
        n_tokens = results["class_token_counts"].get(str(c), results["class_token_counts"].get(c, 0))
        if sf:
            print(f"{label:>15s} | {n_cands:>7d} {n_tokens:>8d} | "
                  f"{sf['feature_idx']:>8d} {sf['kl']:>8.4f} | "
                  f"{sf['precision']:>8.4f} {sf['recall']:>8.4f}")
        else:
            print(f"{label:>15s} | {n_cands:>7d} {n_tokens:>8d} | {'N/A':>8s} {'N/A':>8s} | "
                  f"{'N/A':>8s} {'N/A':>8s}")


# ── Main evaluation loop ─────────────────────────────────────────────────────

def run_verification(
    config: InfoTheoryEvalConfig,
    selected_saes: list[tuple[str, str]],
    device: str,
    hkl_results_path: str,
    output_path: str,
    artifacts_path: str,
    k_values: list[int],
    n_random_trials: int = 10,
    min_density: float = 1e-3,
    max_density: float = float("inf"),
    force_rerun: bool = False,
):
    """Main loop: for each SAE, load H/KL results and compute P/R verification."""

    print(f"[VERIFY] Config: model={config.model_name}, dataset={config.dataset_name}")
    print(f"[VERIFY] k_values={k_values}, n_random_trials={n_random_trials}, "
          f"min_density={min_density}, max_density={max_density}")
    print(f"[VERIFY] H/KL results from: {hkl_results_path}")
    print(f"[VERIFY] Output to: {output_path}")
    print(f"[VERIFY] Total SAEs: {len(selected_saes)}")

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(artifacts_path, exist_ok=True)

    # Load LLM
    llm_dtype = general_utils.str_to_dtype(config.llm_dtype)
    print(f"\n[VERIFY] Loading model {config.model_name} with dtype={llm_dtype}...")
    model = HookedTransformer.from_pretrained_no_processing(
        config.model_name, device=device, dtype=llm_dtype
    )

    # Build label mapping
    label2id = build_label2id(include_non_entity=config.include_non_entity)
    id2label = {v: k for k, v in label2id.items()}
    num_classes = len(label2id)
    print(f"[VERIFY] {num_classes} classes: {list(label2id.keys())}")

    # Cache for LLM activations (reuse across SAEs on same layer + hook)
    cached_layer = None
    cached_hook_name = None
    all_acts_BLD = None
    token_labels_BL = None

    all_results = {}

    for sae_release, sae_id_or_obj in tqdm(selected_saes, desc="SAE Verification"):
        loaded = general_utils.load_and_format_sae(
            sae_release, sae_id_or_obj, device
        )
        assert loaded is not None, f"Failed to load SAE: {sae_release}/{sae_id_or_obj}"
        sae_id, sae, _ = loaded
        sae = sae.to(device=device, dtype=llm_dtype)

        layer = sae.cfg.hook_layer
        hook_name = sae.cfg.hook_name
        sae_key = f"{sae_release}_{sae_id}".replace("/", "_")

        print(f"\n{'='*70}")
        print(f"[VERIFY] === {sae_key} (layer={layer}) ===")

        # Check if output already exists
        out_file = os.path.join(output_path, f"{sae_key}_topk_pr_results.json")
        if os.path.exists(out_file) and not force_rerun:
            print(f"[VERIFY] Skipping (results exist): {out_file}")
            del sae
            continue

        # Load corresponding H/KL result
        hkl_file = os.path.join(hkl_results_path, f"{sae_key}_eval_results.json")
        if not os.path.exists(hkl_file):
            print(f"[WARN] H/KL result not found: {hkl_file}, skipping")
            del sae
            continue
        hkl_result = load_hkl_results(hkl_file)
        hkl_max_density = hkl_result["eval_config"]["max_feature_density"]

        # Build candidate pool: use the verify-time override (default: inf, i.e.
        # no upper bound) instead of the H/KL eval-time max_feature_density.
        # The H/KL eval's max_feature_density only affects the SAE-level mean
        # KL/H aggregate; per-feature (density, KL) are stored unfiltered.
        pool_info = build_topk_candidates(hkl_result, max_density)

        # Load / cache LLM activations
        if layer != cached_layer or hook_name != cached_hook_name:
            ne_tag = "noO" if not config.include_non_entity else "withO"
            ds_short = config.dataset_name.split("/")[-1]
            ds_split_tag = f"{ds_short}_{config.dataset_split}_n{config.num_samples}_ctx{config.context_length}_{ne_tag}"
            cache_artifacts = os.path.join(artifacts_path, config.model_name, ds_split_tag)
            os.makedirs(cache_artifacts, exist_ok=True)

            cache_path = _get_token_acts_cache_path(
                cache_artifacts, config.num_samples, config.context_length,
                layer, hook_name, config.include_non_entity,
            )

            if os.path.exists(cache_path):
                print(f"[VERIFY] Loading cached activations: {cache_path}")
                cache_data = torch.load(cache_path, map_location="cpu", weights_only=False)
                all_acts_BLD = cache_data["acts"]
                token_labels_BL = cache_data["token_labels"]
            else:
                print(f"[VERIFY] Computing activations for layer {layer}...")
                all_acts_BLD, token_labels_BL, _, _ = get_token_level_activations(
                    model, config, layer, hook_name, device
                )
                torch.save(
                    {"acts": all_acts_BLD, "token_labels": token_labels_BL,
                     "label2id": label2id, "num_classes": num_classes},
                    cache_path,
                )
                print(f"[VERIFY] Saved cache: {cache_path}")

            cached_layer = layer
            cached_hook_name = hook_name

        assert all_acts_BLD is not None and token_labels_BL is not None

        # Step A: First SAE encode pass — compute class_acts for feature→class assignment
        # (We need argmax P(c|j) before we can select top-k, so this must come first)
        print(f"[VERIFY] Pass 1: Computing class_acts for feature→class assignment...")
        class_acts, _, _, _ = encode_and_accumulate(
            sae, all_acts_BLD, token_labels_BL,
            num_classes, config.sae_batch_size, device,
        )

        # Step B: Assign features to classes and select top-k for all 4 rankings
        selection = select_topk_from_class_acts(
            class_acts, pool_info["candidates"], pool_info["feature_kl"],
            pool_info["feature_density"], k_values, num_classes, id2label,
            min_density=min_density,
        )

        # Step C: Second SAE encode pass — compute P/R with known top-k sets
        pr_results = compute_precision_recall_streaming(
            sae, all_acts_BLD, token_labels_BL,
            selection["feature_class"],
            selection["topk_per_class"],
            selection["density_topk_per_class"],
            selection["kl_floor_topk_per_class"],
            selection["mi_topk_per_class"],
            selection["class_candidate_counts"],
            selection["class_floor_counts"],
            pool_info["feature_kl"],
            num_classes, k_values,
            config.sae_batch_size, device,
            n_random_trials=n_random_trials,
            random_seed=config.random_seed,
        )

        # Add metadata
        pr_results["sae_release"] = sae_release
        pr_results["sae_id"] = sae_id
        pr_results["layer"] = layer
        pr_results["config"] = {
            "model_name": config.model_name,
            "dataset_name": config.dataset_name,
            "dataset_split": config.dataset_split,
            "num_samples": config.num_samples,
            "context_length": config.context_length,
            "include_non_entity": config.include_non_entity,
            "max_feature_density_hkl_eval": hkl_max_density,
            "max_feature_density_verify": max_density,
            "min_feature_density_floor": min_density,
        }
        pr_results["id2label"] = {str(k): v for k, v in id2label.items()}
        pr_results["datetime_epoch_millis"] = int(time.time() * 1000)

        # Convert dict keys to strings for JSON
        pr_results["per_class"] = {str(k): v for k, v in pr_results["per_class"].items()}
        pr_results["class_token_counts"] = {str(k): v for k, v in pr_results["class_token_counts"].items()}
        pr_results["macro"] = {str(k): v for k, v in pr_results["macro"].items()}

        # Print summary
        print_results_summary(pr_results, id2label)

        # Save
        with open(out_file, "w") as f:
            json.dump(pr_results, f, indent=2)
        print(f"\n[VERIFY] Results saved to {out_file}")

        all_results[sae_key] = pr_results

        del sae
        gc.collect()
        torch.cuda.empty_cache()

    # ── Cross-SAE summary ────────────────────────────────────────────────────
    if all_results:
        print("\n" + "=" * 80)
        print("CROSS-SAE SUMMARY (Recall, all groups)")
        print("=" * 80)
        header = (f"{'SAE':>50s} | {'k':>3s} | "
                  f"{'KL-R':>7s} {'Flr-R':>7s} {'MI-R':>7s} {'Den-R':>7s} {'Rnd-R':>7s} | "
                  f"{'KL-P':>7s} {'Flr-P':>7s} {'MI-P':>7s}")
        print(header)
        print("-" * len(header))
        for sae_key, res in sorted(all_results.items()):
            for k in k_values:
                m = res["macro"].get(str(k), res["macro"].get(k, {}))
                print(f"{sae_key:>50s} | {k:>3d} | "
                      f"{m.get('kl_top_recall', 0):>7.4f} "
                      f"{m.get('kl_floor_top_recall', 0):>7.4f} "
                      f"{m.get('mi_top_recall', 0):>7.4f} "
                      f"{m.get('density_top_recall', 0):>7.4f} "
                      f"{m.get('random_recall', 0):>7.4f} | "
                      f"{m.get('kl_top_precision', 0):>7.4f} "
                      f"{m.get('kl_floor_top_precision', 0):>7.4f} "
                      f"{m.get('mi_top_precision', 0):>7.4f}")

    return all_results


# ── CLI ──────────────────────────────────────────────────────────────────────

def arg_parser():
    parser = argparse.ArgumentParser(
        description="Top-k Feature P/R Verification for H/KL Metrics (PII token-level)"
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--sae_regex_pattern", type=str, required=True)
    parser.add_argument("--sae_block_pattern", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="ai4privacy/pii-masking-300k")
    parser.add_argument("--dataset_split", type=str, default="validation")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--k_values", type=int, nargs="+", default=[1, 5, 10, 20],
                        help="k values for top-k feature selection (default: 1 5 10 20)")
    parser.add_argument("--n_random_trials", type=int, default=10,
                        help="Number of random baseline trials (default: 10)")
    parser.add_argument("--min_density", type=float, default=1e-3,
                        help="Support floor for KL+floor ranking (default: 1e-3). "
                             "Features with density < this are excluded from the "
                             "KL+floor variant only; other rankings are unaffected.")
    parser.add_argument("--max_density", type=float, default=float("inf"),
                        help="Verify-time density ceiling for the candidate pool "
                             "(default: inf, i.e. no upper bound). Overrides the "
                             "H/KL eval's max_feature_density, which only affected "
                             "the SAE-level mean aggregate, not per-feature data.")
    parser.add_argument("--hkl_results_path", type=str, required=True,
                        help="Path to directory containing H/KL eval result JSONs")
    parser.add_argument("--output_folder", type=str, default="eval_results/topk_pr_verification_v2")
    parser.add_argument("--artifacts_path", type=str, default="artifacts/info_theory")
    parser.add_argument("--llm_batch_size", type=int, default=None)
    parser.add_argument("--llm_dtype", type=str, default=None)
    parser.add_argument("--force_rerun", action="store_true")
    return parser


if __name__ == "__main__":
    args = arg_parser().parse_args()
    device = general_utils.setup_environment()
    print(f"[VERIFY] Device: {device}")

    config = InfoTheoryEvalConfig(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        text_column="source_text",
        label_column="privacy_mask",
        num_samples=args.num_samples,
        label_type="token",
        include_non_entity=False,  # noO mode
    )
    if args.llm_batch_size is not None:
        config.llm_batch_size = args.llm_batch_size
    else:
        config.llm_batch_size = activation_collection.LLM_NAME_TO_BATCH_SIZE.get(
            config.model_name, 32
        )
    if args.llm_dtype is not None:
        config.llm_dtype = args.llm_dtype
    else:
        config.llm_dtype = activation_collection.LLM_NAME_TO_DTYPE.get(
            config.model_name, "bfloat16"
        )

    selected_saes = get_saes_from_regex(args.sae_regex_pattern, args.sae_block_pattern)
    assert len(selected_saes) > 0, "No SAEs selected"
    print(f"[VERIFY] Selected {len(selected_saes)} SAE(s)")

    run_verification(
        config=config,
        selected_saes=selected_saes,
        device=device,
        hkl_results_path=args.hkl_results_path,
        output_path=args.output_folder,
        artifacts_path=args.artifacts_path,
        k_values=args.k_values,
        n_random_trials=args.n_random_trials,
        min_density=args.min_density,
        max_density=args.max_density,
        force_rerun=args.force_rerun,
    )
