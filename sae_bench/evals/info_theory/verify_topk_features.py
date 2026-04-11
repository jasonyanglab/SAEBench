"""Top-k Feature Precision/Recall Verification for H/KL Metrics.

Validates that features ranked highly by KL-type scores actually encode their
assigned concepts, via token-level and span-level Precision and Recall.

Pipeline:
    1. Load step-1 H/KL results -> per-feature (KL, density, H_norm)
    2. SAE encode pass A -> class_acts -> argmax P(c|j) -> feature->class
    3. For each class, build ranked lists (see RANKING_GROUPS) and take top-k
    4. SAE encode pass B -> stream per-feature TP/FP/FN and OR-joint per group
       (both token-level and span-level)
    5. Aggregate per-class and macro, emit JSON + summary table

Span-level evaluation:
    A span is a maximal run of consecutive tokens sharing the same class label.
    Span TP: the feature set activates on at least one token of a class-c span.
    Span FP: the feature set activates on at least one token of a non-class-c span.
    Span Recall measures entity discovery (did we find this PII instance?),
    which is more forgiving than token-level for multi-token entities.

Why Recall is the primary metric:
    P(c|j) is used to compute KL/H (Feature->Concept direction). Recall is the
    orthogonal Concept->Feature direction, so it is non-circular.

Why frequency P/R (not amplitude):
    Amplitude precision == P(c|j) -> circular with KL. Counting activation
    events is partially independent of the scoring process.
"""

import argparse
import gc
import json
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
    encode_and_accumulate,
    get_token_level_activations,
    _get_token_acts_cache_path,
    PII_ENTITY_TYPES,
)


# ── Ranking group definitions ────────────────────────────────────────────────
#
# Each verify run evaluates these seven ranked lists per class + a Random
# baseline. The spec drives candidate building, top-k selection, streaming
# TP/FP counters, result assembly and the summary table — add/remove groups
# here and downstream code adapts automatically.
#
#   (group_id, display, descending)
#     group_id    used as metric prefix in JSON output ({group_id}_top_*)
#     display     short header label for the summary table
#     descending  sort order on the score ("kl"/"density"/"mi" desc, H asc)
#
# Layer 1 — baselines (no filter, compare ranking metrics):
#   kl, h, density, mi
# Layer 2 — main method (floor + H ceiling, KL ranking):
#   kl_fh
# Layer 3 — ablations (each removes one component from kl_fh):
#   kl_f  (no H ceiling), h_f (H ranking instead of KL)
RANKING_GROUPS: list[tuple[str, str, bool]] = [
    ("kl",       "KL",    True),   # baseline: raw KL, no filter
    ("h",        "H",     False),  # baseline: raw H ascending, no filter
    ("density",  "Den",   True),   # baseline: frequency
    ("mi",       "MI",    True),   # baseline: density * KL
    ("kl_fh",   "KL_FH", True),   # main: KL, gated by density floor AND H <= max_h
    ("kl_f",    "KL_F",  True),   # ablation: KL, gated by density floor only
    ("h_f",     "H_F",   False),  # ablation: H ascending, gated by density floor
]
GROUP_NAMES: list[str] = [g[0] for g in RANKING_GROUPS]
GROUP_DISPLAY: dict[str, str] = {g[0]: g[1] for g in RANKING_GROUPS}
GROUP_DESCENDING: dict[str, bool] = {g[0]: g[2] for g in RANKING_GROUPS}


def _safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float]:
    """Return (precision, recall) from TP/FP/FN counts."""
    return _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)


def _build_span_info(
    token_labels_BL: torch.Tensor,
    dropped_class_ids: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """Build span instance IDs by detecting consecutive same-label runs.

    A span is a maximal run of consecutive tokens sharing the same class label.
    Tokens with label < 0 or in dropped_class_ids break runs and are assigned
    span_id = 0 (no span).

    Returns:
        span_ids_BL:  [B, L] int64 array. 0 = no span, 1+ = span instance ID.
        span_class_arr: int64 array indexed by span_id → class_id (index 0 unused).
        class_span_count: {class_id: number_of_spans}.
    """
    if dropped_class_ids is None:
        dropped_class_ids = set()
    labels = token_labels_BL.numpy()
    B, L = labels.shape
    span_ids = np.zeros((B, L), dtype=np.int64)
    span_classes: list[int] = [0]  # index 0 unused; span IDs are 1-based
    class_span_count: dict[int, int] = defaultdict(int)

    next_id = 1
    for b in range(B):
        prev = -1
        for t in range(L):
            lab = int(labels[b, t])
            if lab < 0 or lab in dropped_class_ids:
                prev = -1
                continue
            if lab != prev:
                # Start a new span
                span_ids[b, t] = next_id
                span_classes.append(lab)
                class_span_count[lab] += 1
                next_id += 1
            else:
                span_ids[b, t] = next_id - 1
            prev = lab

    span_class_arr = np.array(span_classes, dtype=np.int64)
    return span_ids, span_class_arr, dict(class_span_count)


# ── Step 1: Load H/KL results and build candidate pool ──────────────────────


def load_hkl_results(result_path: str) -> dict:
    with open(result_path, "r") as f:
        return json.load(f)


def build_topk_candidates(hkl_result: dict, max_feature_density: float) -> dict:
    """Build candidate feature pool from step-1 H/KL results.

    A candidate is an alive feature (KL >= 0) with density <= max_feature_density.
    Class assignment and top-k selection are deferred to select_topk_from_class_acts,
    which needs class_acts (from an SAE encode pass) to compute argmax P(c|j).

    Returns a dict with:
        num_classes, candidates (set), feature_kl, feature_density, feature_h, F
    """
    details = hkl_result["eval_result_details"]

    feature_kl: dict[int, float] = {}
    feature_density: dict[int, float] = {}
    feature_h: dict[int, float] = {}
    for d in details:
        idx = d["feature_index"]
        feature_kl[idx] = d["kl_divergence"]
        feature_density[idx] = d["density"]
        feature_h[idx] = d.get("normalized_entropy", -1.0)

    F = len(details)

    include_non_entity = hkl_result["eval_config"].get("include_non_entity", False)
    num_classes = len(PII_ENTITY_TYPES) + (1 if include_non_entity else 0)

    candidates = {
        idx for idx in range(F)
        if feature_kl.get(idx, -1.0) >= 0 and feature_density.get(idx, 0.0) <= max_feature_density
    }

    print(f"[TOPK] Candidate pool: {len(candidates)}/{F} features "
          f"(alive & density <= {max_feature_density})")

    return {
        "num_classes": num_classes,
        "candidates": candidates,
        "feature_kl": feature_kl,
        "feature_density": feature_density,
        "feature_h": feature_h,
        "F": F,
    }


# ── Step 2: Assign features to classes and build top-k per group ─────────────


def _score_feature_for_groups(
    kl_j: float, d_j: float, h_j: float,
    min_density: float, max_h: float,
) -> dict[str, float]:
    """Return {group_name: score} only for groups this feature passes the filter for.

    Filter semantics:
      kl, h, density, mi : no filter (Layer-1 baselines)
      kl_fh              : density >= min_density AND H_norm <= max_h (Layer-2 main)
      kl_f               : density >= min_density (Layer-3 ablation)
      h_f                : density >= min_density (Layer-3 ablation, ranked by H)
    """
    out: dict[str, float] = {
        "kl": kl_j,
        "density": d_j,
        "mi": d_j * kl_j,
    }
    # H baseline needs valid H (h_j >= 0)
    if h_j >= 0.0:
        out["h"] = h_j
    # Floor-gated groups
    if d_j >= min_density:
        out["kl_f"] = kl_j
        if h_j >= 0.0:
            out["h_f"] = h_j
            if h_j <= max_h:
                out["kl_fh"] = kl_j
    return out


def select_topk_from_class_acts(
    class_acts: np.ndarray,
    candidates: set,
    feature_kl: dict,
    feature_density: dict,
    feature_h: dict,
    k_values: list[int],
    num_classes: int,
    id2label: dict[int, str],
    min_density: float = 0.0,
    max_h: float = 1.0,
    dropped_class_ids: set | None = None,
) -> dict:
    """Assign each candidate to c_j = argmax P(c|j), then build top-k per group.

    Features whose c_j lands in dropped_class_ids are excluded entirely — neither
    evaluated nor allowed to contribute to any other class's ranking.

    Returns:
        {
          'feature_class': {feat_idx: class_id},
          'topk':         {group: {c: {k: [feat_idx, ...]}}},
          'class_counts': {group: {c: int}},   # candidates per group per class
        }
    """
    if dropped_class_ids is None:
        dropped_class_ids = set()

    total_act = class_acts.sum(axis=0)  # [F]

    # class_candidates[group][c] -> list of (score, feat_idx); sorted later
    class_candidates: dict[str, defaultdict] = {
        g: defaultdict(list) for g in GROUP_NAMES
    }
    feature_class: dict[int, int] = {}

    for j in candidates:
        if total_act[j] < 1e-10:
            continue
        c_j = int(np.argmax(class_acts[:, j] / total_act[j]))
        if c_j in dropped_class_ids:
            continue
        feature_class[j] = c_j

        scores = _score_feature_for_groups(
            feature_kl[j], feature_density.get(j, 0.0), feature_h.get(j, -1.0),
            min_density, max_h,
        )
        for g, s in scores.items():
            class_candidates[g][c_j].append((s, j))

    # Sort per group / per class
    for g in GROUP_NAMES:
        rev = GROUP_DESCENDING[g]
        for c in class_candidates[g]:
            class_candidates[g][c].sort(key=lambda x: x[0], reverse=rev)

    # Materialize topk[group][c][k] and count tables
    topk: dict[str, dict[int, dict[int, list[int]]]] = {g: {} for g in GROUP_NAMES}
    class_counts: dict[str, dict[int, int]] = {g: {} for g in GROUP_NAMES}

    print(f"\n[TOPK] Feature->class assignment & top-k selection "
          f"(min_density={min_density}, max_h={max_h}, dropped={sorted(dropped_class_ids)}):")
    for c in range(num_classes):
        if c in dropped_class_ids:
            for g in GROUP_NAMES:
                topk[g][c] = {k: [] for k in k_values}
                class_counts[g][c] = 0
            continue
        for g in GROUP_NAMES:
            cands = class_candidates[g].get(c, [])
            class_counts[g][c] = len(cands)
            topk[g][c] = {k: [idx for _, idx in cands[:k]] for k in k_values}

        label = id2label.get(c, f"class_{c}")
        top1_info = " ".join(
            f"{GROUP_DISPLAY[g]}={class_candidates[g][c][0][0]:.4f}"
            if class_candidates[g].get(c) else f"{GROUP_DISPLAY[g]}=N/A"
            for g in GROUP_NAMES
        )
        n_kl = class_counts["kl"][c]
        n_floor = class_counts["kl_f"][c]
        n_floor_h = class_counts["kl_fh"][c]
        print(f"[TOPK]   {label:15s}: {n_kl:5d} cands ({n_floor:4d} floor, "
              f"{n_floor_h:4d} floor+H)  top1[{top1_info}]")

    return {
        "feature_class": feature_class,
        "topk": topk,
        "class_counts": class_counts,
    }


# ── Step 3: Streaming P/R computation ───────────────────────────────────────


@torch.no_grad()
def compute_precision_recall_streaming(
    sae: SAE,
    all_acts_BLD: torch.Tensor,
    token_labels_BL: torch.Tensor,
    feature_class: dict[int, int],
    topk: dict[str, dict[int, dict[int, list[int]]]],
    class_counts: dict[str, dict[int, int]],
    feature_kl: dict[int, float],
    feature_h: dict[int, float],
    num_classes: int,
    k_values: list[int],
    sae_batch_size: int,
    device: str,
    dropped_class_ids: set | None = None,
    n_random_trials: int = 10,
    random_seed: int = 42,
    span_ids_BL: np.ndarray | None = None,
    span_class_arr: np.ndarray | None = None,
    class_span_count: dict[int, int] | None = None,
) -> dict:
    """Stream SAE encode and compute frequency-based P/R for every ranking group.

    Single pass over the data:
      - per-feature TP/FP across classes (drives single-feature P/R at k=1)
      - per-group OR-joint TP/FP for each (class, k)  [token-level]
      - per-group OR-joint span TP/FP for each (class, k)  [span-level]
      - Random baseline (n_random_trials averaged, sampled from candidate pool)

    Token-level: one activation event per token.
    Span-level: a span is hit if ANY of its tokens is activated by the feature set.
    """
    if dropped_class_ids is None:
        dropped_class_ids = set()

    B = all_acts_BLD.shape[0]

    # Random baseline: sample from all candidates assigned to each class
    class_all_features: dict[int, list[int]] = defaultdict(list)
    for fi, c in feature_class.items():
        class_all_features[c].append(fi)

    # ── Build the set of features we need to encode & track ─────────────────

    def _iter_evaluated_classes():
        for c in range(num_classes):
            if c not in dropped_class_ids:
                yield c

    tracked_set: set[int] = set()
    for g in GROUP_NAMES:
        for c in _iter_evaluated_classes():
            for k in k_values:
                tracked_set.update(topk[g][c][k])
    for c in _iter_evaluated_classes():
        tracked_set.update(class_all_features.get(c, []))
    tracked_features = sorted(tracked_set)
    feat_to_local = {f: i for i, f in enumerate(tracked_features)}
    n_tracked = len(tracked_features)

    print(f"\n[P/R] Tracking {n_tracked} features for P/R computation")
    print(f"[P/R] Streaming SAE encode over {B} samples, batch_size={sae_batch_size}")

    # Per-feature counters (single-feature P/R)
    feat_class_act_count = np.zeros((n_tracked, num_classes), dtype=np.int64)
    feat_act_count = np.zeros(n_tracked, dtype=np.int64)
    class_token_count = np.zeros(num_classes, dtype=np.int64)

    # Joint OR-union counters: group_tp[group][c][k]
    def _zero_ckdict():
        return {c: {k: 0 for k in k_values} for c in range(num_classes)}
    group_tp = {g: _zero_ckdict() for g in GROUP_NAMES}
    group_fp = {g: _zero_ckdict() for g in GROUP_NAMES}

    # Pre-computed local-index arrays: group_local[group][(c,k)] -> np.array | None
    def _to_local(feats: list[int]):
        return np.array([feat_to_local[f] for f in feats]) if feats else None
    group_local: dict[str, dict[tuple[int, int], np.ndarray | None]] = {
        g: {(c, k): _to_local(topk[g][c][k])
            for c in range(num_classes) for k in k_values}
        for g in GROUP_NAMES
    }

    # Random baseline: pre-sample feature sets per (c, k, trial)
    rng = np.random.RandomState(random_seed)
    random_local: dict[tuple[int, int, int], np.ndarray | None] = {}
    for c in range(num_classes):
        pool_c = class_all_features.get(c, [])
        for k in k_values:
            for trial in range(n_random_trials):
                n_sample = min(k, len(pool_c))
                if n_sample > 0:
                    sampled = rng.choice(pool_c, size=n_sample, replace=False)
                    random_local[(c, k, trial)] = np.array([feat_to_local[f] for f in sampled])
                else:
                    random_local[(c, k, trial)] = None
    random_tp = {c: {k: np.zeros(n_random_trials, dtype=np.int64) for k in k_values}
                 for c in range(num_classes)}
    random_fp = {c: {k: np.zeros(n_random_trials, dtype=np.int64) for k in k_values}
                 for c in range(num_classes)}

    # Span-level counters (parallel to token-level group_tp/group_fp)
    do_span = span_ids_BL is not None
    span_ids_tensor = torch.from_numpy(span_ids_BL) if do_span else None
    span_group_tp = {g: _zero_ckdict() for g in GROUP_NAMES}
    span_group_fp = {g: _zero_ckdict() for g in GROUP_NAMES}
    span_random_tp = {c: {k: np.zeros(n_random_trials, dtype=np.int64) for k in k_values}
                      for c in range(num_classes)}
    span_random_fp = {c: {k: np.zeros(n_random_trials, dtype=np.int64) for k in k_values}
                      for c in range(num_classes)}

    tracked_indices_tensor = torch.tensor(tracked_features, dtype=torch.long, device=device)

    # ── Single streaming pass ────────────────────────────────────────────────

    for i in tqdm(range(0, B, sae_batch_size), desc="SAE Encode (P/R)"):
        batch_acts = all_acts_BLD[i : i + sae_batch_size].to(device)
        batch_sae = sae.encode(batch_acts)  # [b, L, F]
        batch_labels = token_labels_BL[i : i + sae_batch_size]

        # Project to only tracked features to keep memory bounded
        batch_tracked = batch_sae[:, :, tracked_indices_tensor].cpu()
        flat_tracked = batch_tracked.reshape(-1, n_tracked)
        flat_labels = batch_labels.reshape(-1)
        flat_span_ids: torch.Tensor | None = None
        if do_span:
            assert span_ids_tensor is not None
            flat_span_ids = span_ids_tensor[i : i + sae_batch_size].reshape(-1)
        del batch_acts, batch_sae, batch_tracked

        # Valid = labelled token AND not belonging to a dropped class
        valid_mask = flat_labels >= 0
        for dc in dropped_class_ids:
            valid_mask &= (flat_labels != dc)
        if not valid_mask.any():
            continue

        v_tracked = flat_tracked[valid_mask].float().numpy()  # [V, n_tracked]
        v_labels = flat_labels[valid_mask].numpy()             # [V]
        v_span_ids: np.ndarray | None = None
        if do_span:
            assert flat_span_ids is not None
            v_span_ids = flat_span_ids[valid_mask].numpy()  # [V]
        v_active = v_tracked > 0                               # [V, n_tracked] bool
        del flat_tracked, flat_labels, flat_span_ids

        for c in _iter_evaluated_classes():
            cmask = v_labels == c
            not_cmask = ~cmask
            n_c = int(cmask.sum())
            if n_c > 0:
                class_token_count[c] += n_c
                feat_class_act_count[:, c] += v_active[cmask].sum(axis=0).astype(np.int64)

            for k in k_values:
                # Ranked groups
                for g in GROUP_NAMES:
                    idx = group_local[g][(c, k)]
                    if idx is None:
                        continue
                    joint_active = v_active[:, idx].any(axis=1)
                    group_tp[g][c][k] += int((joint_active & cmask).sum())
                    group_fp[g][c][k] += int((joint_active & not_cmask).sum())
                    # Span-level: aggregate per-token hits to per-span hits
                    if do_span:
                        assert v_span_ids is not None and span_class_arr is not None
                        hit_sids = np.unique(v_span_ids[joint_active & (v_span_ids > 0)])
                        if len(hit_sids) > 0:
                            sc = span_class_arr[hit_sids]
                            span_group_tp[g][c][k] += int((sc == c).sum())
                            span_group_fp[g][c][k] += int((sc != c).sum())

                # Random baseline
                for trial in range(n_random_trials):
                    rl = random_local[(c, k, trial)]
                    if rl is None:
                        continue
                    rand_active = v_active[:, rl].any(axis=1)
                    random_tp[c][k][trial] += int((rand_active & cmask).sum())
                    random_fp[c][k][trial] += int((rand_active & not_cmask).sum())
                    if do_span:
                        assert v_span_ids is not None and span_class_arr is not None
                        hit_sids = np.unique(v_span_ids[rand_active & (v_span_ids > 0)])
                        if len(hit_sids) > 0:
                            sc = span_class_arr[hit_sids]
                            span_random_tp[c][k][trial] += int((sc == c).sum())
                            span_random_fp[c][k][trial] += int((sc != c).sum())

        feat_act_count += v_active.sum(axis=0).astype(np.int64)
        del v_tracked, v_labels, v_active

    print(f"[P/R] Streaming done. Class token counts: {class_token_count.tolist()}")

    # ── Assemble per-class results ───────────────────────────────────────────

    results: dict = {
        "k_values": k_values,
        "n_random_trials": n_random_trials,
        "class_token_counts": {int(c): int(class_token_count[c]) for c in range(num_classes)},
        "class_span_counts": class_span_count if do_span else {},
        "per_class": {},
        "macro": {},
    }

    # Single-feature (KL_FH top-1) P/R per class
    single_feature_results: dict[int, dict] = {}
    for c in _iter_evaluated_classes():
        top1 = topk["kl_fh"][c].get(1, [])
        if not top1:
            continue
        li = feat_to_local[top1[0]]
        tp = int(feat_class_act_count[li, c])
        fp = int(feat_act_count[li]) - tp
        fn = int(class_token_count[c]) - tp
        p, r = _prf(tp, fp, fn)
        single_feature_results[c] = {
            "feature_idx": top1[0],
            "kl": feature_kl[top1[0]],
            "h": feature_h.get(top1[0], -1.0),
            "precision": p,
            "recall": r,
        }

    # Per-class joint results
    for c in range(num_classes):
        if c in dropped_class_ids:
            results["per_class"][c] = {
                "dropped": True,
                "n_candidates": {g: 0 for g in GROUP_NAMES},
                "single_feature": {},
                "joint": {},
            }
            continue

        n_cls_tok = int(class_token_count[c])
        per_k: dict[int, dict] = {}
        for k in k_values:
            entry: dict = {}
            n_cls_spans = class_span_count[c] if do_span and class_span_count else 0
            for g in GROUP_NAMES:
                n_feats = len(topk[g][c][k])
                # Token-level P/R
                tp = group_tp[g][c][k]
                fp = group_fp[g][c][k]
                p, r = _prf(tp, fp, n_cls_tok - tp)
                entry[f"n_features_used_{g}"] = n_feats
                entry[f"{g}_top_precision"] = p
                entry[f"{g}_top_recall"] = r
                # Span-level P/R
                if do_span:
                    stp = span_group_tp[g][c][k]
                    sfp = span_group_fp[g][c][k]
                    sp, sr = _prf(stp, sfp, n_cls_spans - stp)
                    entry[f"{g}_span_precision"] = sp
                    entry[f"{g}_span_recall"] = sr

            # Random baseline (averaged over trials)
            rp_list, rr_list = [], []
            srp_list, srr_list = [], []
            for trial in range(n_random_trials):
                rtp = int(random_tp[c][k][trial])
                rfp = int(random_fp[c][k][trial])
                p, r = _prf(rtp, rfp, n_cls_tok - rtp)
                rp_list.append(p)
                rr_list.append(r)
                if do_span:
                    srtp = int(span_random_tp[c][k][trial])
                    srfp = int(span_random_fp[c][k][trial])
                    sp, sr = _prf(srtp, srfp, n_cls_spans - srtp)
                    srp_list.append(sp)
                    srr_list.append(sr)
            entry["random_precision_mean"] = float(np.mean(rp_list)) if rp_list else 0.0
            entry["random_precision_std"] = float(np.std(rp_list)) if rp_list else 0.0
            entry["random_recall_mean"] = float(np.mean(rr_list)) if rr_list else 0.0
            entry["random_recall_std"] = float(np.std(rr_list)) if rr_list else 0.0
            if do_span:
                entry["random_span_precision_mean"] = float(np.mean(srp_list)) if srp_list else 0.0
                entry["random_span_recall_mean"] = float(np.mean(srr_list)) if srr_list else 0.0

            per_k[k] = entry

        results["per_class"][c] = {
            "dropped": False,
            "n_candidates": {g: class_counts[g].get(c, 0) for g in GROUP_NAMES},
            "n_spans": class_span_count[c] if do_span and class_span_count else 0,
            "single_feature": single_feature_results.get(c, {}),
            "joint": per_k,
        }

    # ── Macro averages ───────────────────────────────────────────────────────
    #
    # A class contributes to a group's macro only if that group produced at
    # least one feature for the class (avoids biasing e.g. Flr+H downward for
    # classes where the H filter left no candidates).
    for k in k_values:
        m: dict = {}
        per_group_recall: dict[str, list[float]] = {}

        for g in GROUP_NAMES:
            p_list: list[float] = []
            r_list: list[float] = []
            sp_list: list[float] = []
            sr_list: list[float] = []
            for c in _iter_evaluated_classes():
                e = results["per_class"][c]["joint"].get(k, {})
                if e and e.get(f"n_features_used_{g}", 0) > 0:
                    p_list.append(e[f"{g}_top_precision"])
                    r_list.append(e[f"{g}_top_recall"])
                    if do_span and f"{g}_span_recall" in e:
                        sp_list.append(e[f"{g}_span_precision"])
                        sr_list.append(e[f"{g}_span_recall"])
            m[f"{g}_top_precision"] = float(np.mean(p_list)) if p_list else 0.0
            m[f"{g}_top_recall"] = float(np.mean(r_list)) if r_list else 0.0
            if do_span:
                m[f"{g}_span_precision"] = float(np.mean(sp_list)) if sp_list else 0.0
                m[f"{g}_span_recall"] = float(np.mean(sr_list)) if sr_list else 0.0
            m[f"n_classes_evaluated_{g}"] = len(p_list)
            per_group_recall[g] = r_list

        # Random macro: include classes that have at least one KL feature
        rand_p, rand_r = [], []
        srand_p, srand_r = [], []
        for c in _iter_evaluated_classes():
            e = results["per_class"][c]["joint"].get(k, {})
            if e and e.get("n_features_used_kl", 0) > 0:
                rand_p.append(e["random_precision_mean"])
                rand_r.append(e["random_recall_mean"])
                if do_span and "random_span_recall_mean" in e:
                    srand_p.append(e["random_span_precision_mean"])
                    srand_r.append(e["random_span_recall_mean"])
        m["random_precision"] = float(np.mean(rand_p)) if rand_p else 0.0
        m["random_recall"] = float(np.mean(rand_r)) if rand_r else 0.0
        if do_span:
            m["random_span_precision"] = float(np.mean(srand_p)) if srand_p else 0.0
            m["random_span_recall"] = float(np.mean(srand_r)) if srand_r else 0.0

        m["n_classes_evaluated"] = m["n_classes_evaluated_kl"]

        # Auto-generate recall deltas: every group vs kl baseline, plus kl_fh vs kl_f
        def _delta(a: list[float], b: list[float]) -> float:
            return float(np.mean(a) - np.mean(b)) if a and b else 0.0
        m["recall_delta_kl_vs_random"] = _delta(per_group_recall["kl"], rand_r)
        for g in GROUP_NAMES:
            if g != "kl":
                m[f"recall_delta_{g}_vs_kl"] = _delta(per_group_recall[g], per_group_recall["kl"])
        # Key ablation delta: main method vs floor-only
        m["recall_delta_kl_fh_vs_kl_f"] = _delta(per_group_recall["kl_fh"], per_group_recall["kl_f"])

        results["macro"][k] = m

    return results


# ── Pretty-print results ─────────────────────────────────────────────────────


def _macro_for_k(results: dict, k: int) -> dict:
    """Look up macro row tolerating str-keyed macros (after JSON round-trip)."""
    return results["macro"].get(str(k), results["macro"].get(k, {}))


def print_results_summary(results: dict, id2label: dict[int, str]) -> None:
    """Human-readable summary of P/R verification results."""
    print("\n" + "=" * 80)
    print("TOP-K FEATURE PRECISION/RECALL VERIFICATION RESULTS")
    print("=" * 80)

    # ── Macro table: one (P, R) column per group + Rnd ──────────────────────
    print("\n── Macro-Averaged Results (across classes) ──")
    group_cols = GROUP_NAMES + ["random"]
    group_headers = [GROUP_DISPLAY[g] for g in GROUP_NAMES] + ["Rnd"]

    header = f"{'k':>3s} | " + " | ".join(
        f"{h + ' P':>6s} {h + ' R':>6s}" for h in group_headers
    )
    print(header)
    print("-" * len(header))

    def _fmt_pair(m: dict, g: str) -> str:
        if g == "random":
            p, r = m.get("random_precision", 0), m.get("random_recall", 0)
        else:
            p, r = m.get(f"{g}_top_precision", 0), m.get(f"{g}_top_recall", 0)
        return f"{p:>6.3f} {r:>6.3f}"

    for k in results["k_values"]:
        m = _macro_for_k(results, k)
        row = f"{k:>3d} | " + " | ".join(_fmt_pair(m, g) for g in group_cols)
        print(row)

    first_k = results["k_values"][0]
    n_eval = _macro_for_k(results, first_k).get("n_classes_evaluated", 0)
    print(f"\n  ({n_eval} classes evaluated)")

    # ── Span-level macro table (if available) ──────────────────────────────
    first_m = _macro_for_k(results, first_k)
    if first_m.get("kl_span_recall") is not None:
        print("\n── Span-Level Macro-Averaged Results (across classes) ──")

        header_s = f"{'k':>3s} | " + " | ".join(
            f"{h + ' sP':>6s} {h + ' sR':>6s}" for h in group_headers
        )
        print(header_s)
        print("-" * len(header_s))

        def _fmt_span_pair(m: dict, g: str) -> str:
            if g == "random":
                p = m.get("random_span_precision", 0)
                r = m.get("random_span_recall", 0)
            else:
                p = m.get(f"{g}_span_precision", 0)
                r = m.get(f"{g}_span_recall", 0)
            return f"{p:>6.3f} {r:>6.3f}"

        for k in results["k_values"]:
            m = _macro_for_k(results, k)
            row = f"{k:>3d} | " + " | ".join(_fmt_span_pair(m, g) for g in group_cols)
            print(row)

    # ── Per-class k=1 single-feature results (from KL_FH group) ──────────────
    print("\n── Per-Class Single Feature (KL_FH top-1) Results ──")
    print(f"{'Class':>15s} | {'#All':>6s} {'#FH':>5s} {'#Tok':>7s} | "
          f"{'feat':>7s} {'KL':>7s} {'H':>6s} | {'Prec':>7s} {'Recall':>7s}")
    print("-" * 85)
    for c_key, data in sorted(results["per_class"].items(), key=lambda x: int(x[0])):
        c = int(c_key)
        label = id2label.get(c, f"class_{c}")
        if data.get("dropped", False):
            print(f"{label:>15s} | {'DROPPED':>20s} |")
            continue
        sf = data.get("single_feature", {})
        cands = data.get("n_candidates", {})
        n_all = cands.get("kl", 0) if isinstance(cands, dict) else cands
        n_fh = cands.get("kl_fh", 0) if isinstance(cands, dict) else 0
        n_tokens = results["class_token_counts"].get(str(c), results["class_token_counts"].get(c, 0))
        if sf:
            h_str = f"{sf['h']:>6.3f}" if sf.get("h", -1) >= 0 else f"{'N/A':>6s}"
            print(f"{label:>15s} | {n_all:>6d} {n_fh:>5d} {n_tokens:>7d} | "
                  f"{sf['feature_idx']:>7d} {sf['kl']:>7.4f} {h_str} | "
                  f"{sf['precision']:>7.4f} {sf['recall']:>7.4f}")
        else:
            print(f"{label:>15s} | {n_all:>6d} {n_fh:>5d} {n_tokens:>7d} | "
                  f"{'---':>7s} {'---':>7s} {'---':>6s} | {'---':>7s} {'---':>7s}")


def _print_cross_sae_summary(all_results: dict, k_values: list[int]) -> None:
    print("\n" + "=" * 80)
    print("CROSS-SAE SUMMARY (Token-level Recall & Precision, all groups)")
    print("=" * 80)
    r_headers = [f"{GROUP_DISPLAY[g]}-R" for g in GROUP_NAMES] + ["Rnd-R"]
    p_headers = [f"{GROUP_DISPLAY[g]}-P" for g in GROUP_NAMES]
    header = (f"{'SAE':>50s} | {'k':>3s} | "
              + " ".join(f"{h:>7s}" for h in r_headers) + " | "
              + " ".join(f"{h:>7s}" for h in p_headers))
    print(header)
    print("-" * len(header))
    for sae_key, res in sorted(all_results.items()):
        for k in k_values:
            m = _macro_for_k(res, k)
            r_vals = [m.get(f"{g}_top_recall", 0) for g in GROUP_NAMES] + [m.get("random_recall", 0)]
            p_vals = [m.get(f"{g}_top_precision", 0) for g in GROUP_NAMES]
            row = (f"{sae_key:>50s} | {k:>3d} | "
                   + " ".join(f"{v:>7.4f}" for v in r_vals) + " | "
                   + " ".join(f"{v:>7.4f}" for v in p_vals))
            print(row)

    # Span-level summary (if available)
    has_span = any(
        _macro_for_k(res, k_values[0]).get("kl_span_recall") is not None
        for res in all_results.values()
    )
    if has_span:
        print("\n" + "=" * 80)
        print("CROSS-SAE SUMMARY (Span-level Recall & Precision, all groups)")
        print("=" * 80)
        sr_headers = [f"{GROUP_DISPLAY[g]}-sR" for g in GROUP_NAMES] + ["Rnd-sR"]
        sp_headers = [f"{GROUP_DISPLAY[g]}-sP" for g in GROUP_NAMES]
        header_s = (f"{'SAE':>50s} | {'k':>3s} | "
                    + " ".join(f"{h:>7s}" for h in sr_headers) + " | "
                    + " ".join(f"{h:>7s}" for h in sp_headers))
        print(header_s)
        print("-" * len(header_s))
        for sae_key, res in sorted(all_results.items()):
            for k in k_values:
                m = _macro_for_k(res, k)
                sr_vals = [m.get(f"{g}_span_recall", 0) for g in GROUP_NAMES] + [m.get("random_span_recall", 0)]
                sp_vals = [m.get(f"{g}_span_precision", 0) for g in GROUP_NAMES]
                row = (f"{sae_key:>50s} | {k:>3d} | "
                       + " ".join(f"{v:>7.4f}" for v in sr_vals) + " | "
                       + " ".join(f"{v:>7.4f}" for v in sp_vals))
                print(row)


# ── Main evaluation loop ─────────────────────────────────────────────────────


def _load_or_compute_activations(
    model, config: InfoTheoryEvalConfig, layer: int, hook_name: str,
    artifacts_path: str, label2id: dict, num_classes: int, device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (all_acts_BLD, token_labels_BL), using on-disk cache when available."""
    ne_tag = "noO" if not config.include_non_entity else "withO"
    ds_short = config.dataset_name.split("/")[-1]
    ds_split_tag = (f"{ds_short}_{config.dataset_split}_n{config.num_samples}"
                    f"_ctx{config.context_length}_{ne_tag}")
    cache_dir = os.path.join(artifacts_path, config.model_name, ds_split_tag)
    os.makedirs(cache_dir, exist_ok=True)

    cache_path = _get_token_acts_cache_path(
        cache_dir, config.num_samples, config.context_length,
        layer, hook_name, config.include_non_entity,
    )

    if os.path.exists(cache_path):
        print(f"[VERIFY] Loading cached activations: {cache_path}")
        cache_data = torch.load(cache_path, map_location="cpu", weights_only=False)
        return cache_data["acts"], cache_data["token_labels"]

    print(f"[VERIFY] Computing activations for layer {layer}...")
    acts, labels, _, _ = get_token_level_activations(
        model, config, layer, hook_name, device
    )
    torch.save(
        {"acts": acts, "token_labels": labels, "label2id": label2id, "num_classes": num_classes},
        cache_path,
    )
    print(f"[VERIFY] Saved cache: {cache_path}")
    return acts, labels


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
    max_h: float = 0.5,
    drop_classes: list[str] | None = None,
    force_rerun: bool = False,
) -> dict:
    """For each SAE: load its H/KL results and compute top-k P/R verification."""
    if drop_classes is None:
        drop_classes = []

    print(f"[VERIFY] Config: model={config.model_name}, dataset={config.dataset_name}")
    print(f"[VERIFY] k_values={k_values}, n_random_trials={n_random_trials}, "
          f"min_density={min_density}, max_density={max_density}, "
          f"max_h={max_h}, drop_classes={drop_classes}")
    print(f"[VERIFY] H/KL results from: {hkl_results_path}")
    print(f"[VERIFY] Output to: {output_path}")
    print(f"[VERIFY] Total SAEs: {len(selected_saes)}")

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(artifacts_path, exist_ok=True)

    # Load LLM once, reuse across all SAEs
    llm_dtype = general_utils.str_to_dtype(config.llm_dtype)
    print(f"\n[VERIFY] Loading model {config.model_name} with dtype={llm_dtype}...")
    model = HookedTransformer.from_pretrained_no_processing(
        config.model_name, device=device, dtype=llm_dtype
    )

    label2id = build_label2id(include_non_entity=config.include_non_entity)
    id2label = {v: k for k, v in label2id.items()}
    num_classes = len(label2id)
    print(f"[VERIFY] {num_classes} classes: {list(label2id.keys())}")

    dropped_class_ids: set[int] = set()
    for cls_name in drop_classes:
        if cls_name in label2id:
            dropped_class_ids.add(label2id[cls_name])
            print(f"[VERIFY] Dropping class '{cls_name}' (id={label2id[cls_name]}): "
                  f"tokens treated as non-entity, not evaluated")
        else:
            print(f"[VERIFY] WARNING: drop_class '{cls_name}' not in label2id, ignored")

    # Activation cache (reused across SAEs that share layer + hook_name)
    cached_layer: int | None = None
    cached_hook_name: str | None = None
    all_acts_BLD: torch.Tensor | None = None
    token_labels_BL: torch.Tensor | None = None
    span_ids_BL: np.ndarray | None = None
    span_class_arr: np.ndarray | None = None
    class_span_count: dict[int, int] | None = None

    all_results: dict = {}

    for sae_release, sae_id_or_obj in tqdm(selected_saes, desc="SAE Verification"):
        loaded = general_utils.load_and_format_sae(sae_release, sae_id_or_obj, device)
        assert loaded is not None, f"Failed to load SAE: {sae_release}/{sae_id_or_obj}"
        sae_id, sae, _ = loaded
        sae = sae.to(device=device, dtype=llm_dtype)

        layer = sae.cfg.hook_layer
        hook_name = sae.cfg.hook_name
        sae_key = f"{sae_release}_{sae_id}".replace("/", "_")

        print(f"\n{'='*70}")
        print(f"[VERIFY] === {sae_key} (layer={layer}) ===")

        out_file = os.path.join(output_path, f"{sae_key}_topk_pr_results.json")
        if os.path.exists(out_file) and not force_rerun:
            print(f"[VERIFY] Skipping (results exist): {out_file}")
            del sae
            continue

        hkl_file = os.path.join(hkl_results_path, f"{sae_key}_eval_results.json")
        if not os.path.exists(hkl_file):
            print(f"[WARN] H/KL result not found: {hkl_file}, skipping")
            del sae
            continue
        hkl_result = load_hkl_results(hkl_file)
        hkl_max_density = hkl_result["eval_config"]["max_feature_density"]

        # H/KL eval's max_feature_density only affected the SAE-level mean;
        # per-feature (density, KL, H) are stored unfiltered, so we use the
        # verify-time max_density (default: inf = no ceiling).
        pool_info = build_topk_candidates(hkl_result, max_density)

        # Rebuild activation cache if layer or hook changed
        if layer != cached_layer or hook_name != cached_hook_name:
            all_acts_BLD, token_labels_BL = _load_or_compute_activations(
                model, config, layer, hook_name, artifacts_path,
                label2id, num_classes, device,
            )
            # Build span info for span-level P/R
            print(f"[VERIFY] Building span info for span-level evaluation...")
            span_ids_BL, span_class_arr, class_span_count = _build_span_info(
                token_labels_BL, dropped_class_ids,
            )
            n_total_spans = sum(class_span_count.values())
            print(f"[VERIFY] {n_total_spans} spans across {len(class_span_count)} classes")
            cached_layer = layer
            cached_hook_name = hook_name

        assert all_acts_BLD is not None and token_labels_BL is not None

        # Pass A: class_acts for feature->class assignment (argmax P(c|j))
        print(f"[VERIFY] Pass 1: Computing class_acts for feature->class assignment...")
        class_acts, _, _, _ = encode_and_accumulate(
            sae, all_acts_BLD, token_labels_BL,
            num_classes, config.sae_batch_size, device,
        )

        # Build top-k lists for all groups
        selection = select_topk_from_class_acts(
            class_acts, pool_info["candidates"], pool_info["feature_kl"],
            pool_info["feature_density"], pool_info["feature_h"],
            k_values, num_classes, id2label,
            min_density=min_density, max_h=max_h,
            dropped_class_ids=dropped_class_ids,
        )

        # Pass B: stream encode and compute joint + single-feature P/R
        pr_results = compute_precision_recall_streaming(
            sae, all_acts_BLD, token_labels_BL,
            feature_class=selection["feature_class"],
            topk=selection["topk"],
            class_counts=selection["class_counts"],
            feature_kl=pool_info["feature_kl"],
            feature_h=pool_info["feature_h"],
            num_classes=num_classes,
            k_values=k_values,
            sae_batch_size=config.sae_batch_size,
            device=device,
            dropped_class_ids=dropped_class_ids,
            n_random_trials=n_random_trials,
            random_seed=config.random_seed,
            span_ids_BL=span_ids_BL,
            span_class_arr=span_class_arr,
            class_span_count=class_span_count,
        )

        # Attach metadata
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
            "max_h_purity": max_h,
            "drop_classes": drop_classes,
        }
        pr_results["id2label"] = {str(k): v for k, v in id2label.items()}
        pr_results["datetime_epoch_millis"] = int(time.time() * 1000)

        # JSON-serializable keys
        pr_results["per_class"] = {str(k): v for k, v in pr_results["per_class"].items()}
        pr_results["class_token_counts"] = {str(k): v for k, v in pr_results["class_token_counts"].items()}
        pr_results["class_span_counts"] = {str(k): v for k, v in pr_results["class_span_counts"].items()}
        pr_results["macro"] = {str(k): v for k, v in pr_results["macro"].items()}

        print_results_summary(pr_results, id2label)

        with open(out_file, "w") as f:
            json.dump(pr_results, f, indent=2)
        print(f"\n[VERIFY] Results saved to {out_file}")

        all_results[sae_key] = pr_results
        del sae
        gc.collect()
        torch.cuda.empty_cache()

    if all_results:
        _print_cross_sae_summary(all_results, k_values)

    return all_results


# ── CLI ──────────────────────────────────────────────────────────────────────


def arg_parser() -> argparse.ArgumentParser:
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
                        help="Support floor for Flr / Flr+H / H-rank groups. "
                             "Other groups are unaffected.")
    parser.add_argument("--max_density", type=float, default=float("inf"),
                        help="Verify-time density ceiling for the candidate pool "
                             "(default: inf). The step-1 H/KL eval's max_feature_density "
                             "only affected its SAE-level aggregate, not per-feature data.")
    parser.add_argument("--max_h", type=float, default=0.5,
                        help="H_norm purity ceiling for the Flr+H group only "
                             "(default: 0.5).")
    parser.add_argument("--drop_classes", type=str, nargs="*", default=[],
                        help="Class names to exclude entirely (e.g. --drop_classes CARDISSUER). "
                             "Their tokens are treated as non-entity; no top-k is computed for them.")
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
    config.llm_batch_size = args.llm_batch_size if args.llm_batch_size is not None \
        else activation_collection.LLM_NAME_TO_BATCH_SIZE.get(config.model_name, 32)
    config.llm_dtype = args.llm_dtype if args.llm_dtype is not None \
        else activation_collection.LLM_NAME_TO_DTYPE.get(config.model_name, "bfloat16")

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
        max_h=args.max_h,
        drop_classes=args.drop_classes,
        force_rerun=args.force_rerun,
    )
