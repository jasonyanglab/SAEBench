"""验证 SAE 激活值是否严格稀疏（精确零），还是存在接近零的浮点噪声。

背景：
  verify_topk_features.py 用 `v_active = v_tracked > 0` 判定特征是否激活。
  本脚本验证这个判定是否安全——即 SAE encode 输出的非零值是否都远离零。

数据来源：
  main_token.py 产生的 hook 缓存文件（dict with keys: acts, token_labels, ...）

评判标准：
  1. 统计 (0, threshold] 区间的激活数量，threshold 从 1e-10 到 1e-2
     → 任何 threshold 下 count > 0 说明存在极小值
  2. 非零激活值的分位数分布
     → 最小值 >> 1e-4 说明严格稀疏
  3. 最终判定：
     min > 1e-4  → CLEAN，`> 0` 判定正确
     min < 1e-8  → 存在噪声，需要加阈值

用法（远程）：
  python my_scripts/check_sae_activation_sparsity.py
"""

import torch
import numpy as np
from sae_lens import SAE

# ── 配置 ──────────────────────────────────────────────────────────────────────
# main_token.py 产生的 hook 缓存文件
CACHE_PATH = (
    "artifacts/info_theory/gemma-2-2b/"
    "pii-masking-300k_validation_n10000_ctx128_noO/"
    "token_acts_n10000_ctx128_layer12_blocks_12_hook_resid_post_noO.pt"
)
SAE_RELEASE = "gemma-scope-2b-pt-res"
SAE_ID = "layer_12/width_16k/average_l0_82"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLES = 200          # 编码的样本数（足够看清分布）
SAE_BATCH_SIZE = 32
THRESHOLDS = [1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2]


@torch.no_grad()
def main():
    # ── 1. 加载 main_token.py 产生的 hook 缓存 ──────────────────────────────
    print(f"Loading hook cache from: {CACHE_PATH}")
    cache = torch.load(CACHE_PATH, map_location="cpu")
    all_acts_BLD = cache["acts"]
    print(f"  acts shape: {all_acts_BLD.shape}, dtype: {all_acts_BLD.dtype}")
    print(f"  token_labels shape: {cache['token_labels'].shape}")
    print(f"  num_classes: {cache['num_classes']}")

    # ── 2. 加载 SAE 并编码 ──────────────────────────────────────────────────
    print(f"\nLoading SAE: {SAE_RELEASE} / {SAE_ID}")
    sae = SAE.from_pretrained(SAE_RELEASE, SAE_ID, device=DEVICE)[0]
    print(f"  architecture: {sae.cfg.architecture}")
    print(f"  d_sae: {sae.cfg.d_sae}")

    # 分批编码，收集所有非零激活值
    n = min(N_SAMPLES, all_acts_BLD.shape[0])
    all_nonzero: list[np.ndarray] = []
    total_entries = 0
    total_zeros = 0
    total_negative = 0

    print(f"\nEncoding {n} samples in batches of {SAE_BATCH_SIZE} ...")
    for i in range(0, n, SAE_BATCH_SIZE):
        batch = all_acts_BLD[i : i + SAE_BATCH_SIZE].to(device=DEVICE, dtype=sae.dtype)
        sae_out = sae.encode(batch)  # [b, L, d_sae]
        flat = sae_out.float().cpu().numpy().ravel()

        total_entries += len(flat)
        total_zeros += int((flat == 0.0).sum())
        total_negative += int((flat < 0.0).sum())
        nonzero = flat[flat != 0.0]
        if len(nonzero) > 0:
            all_nonzero.append(nonzero)
        del batch, sae_out

    nonzero_vals = np.concatenate(all_nonzero) if all_nonzero else np.array([])
    n_nonzero = len(nonzero_vals)
    pos_vals = nonzero_vals[nonzero_vals > 0]

    # ── 3. 输出分析结果 ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SPARSITY ANALYSIS")
    print(f"{'='*60}")
    print(f"Total entries:      {total_entries:>15,d}")
    print(f"Exact zeros:        {total_zeros:>15,d}  ({total_zeros/total_entries*100:.2f}%)")
    print(f"Non-zero:           {n_nonzero:>15,d}  ({n_nonzero/total_entries*100:.2f}%)")
    print(f"Negative values:    {total_negative:>15,d}")

    if len(pos_vals) == 0:
        print("\nNo positive activations found.")
        return

    # ── 分位数分布 ──
    print(f"\n{'─'*60}")
    print("NON-ZERO POSITIVE ACTIVATION DISTRIBUTION")
    print(f"{'─'*60}")
    pctls = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]
    print(f"  {'Percentile':>12s}  {'Value':>15s}")
    for p in pctls:
        print(f"  {p:>11.1f}%  {np.percentile(pos_vals, p):>15.6e}")
    print(f"  {'min':>12s}  {pos_vals.min():>15.6e}")
    print(f"  {'max':>12s}  {pos_vals.max():>15.6e}")

    # ── 阈值检查 ──
    print(f"\n{'─'*60}")
    print("THRESHOLD CHECK: count of activations in (0, threshold]")
    print(f"{'─'*60}")
    print(f"  {'Threshold':>12s}  {'Count':>10s}  {'% of non-zero':>14s}  {'Status'}")
    for t in THRESHOLDS:
        cnt = int((pos_vals <= t).sum())
        pct = cnt / len(pos_vals) * 100
        status = "!! NOISE" if cnt > 0 else "CLEAN"
        print(f"  {t:>12.0e}  {cnt:>10,d}  {pct:>13.4f}%  {status}")

    # ── 最终判定 ──
    min_val = pos_vals.min()
    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")
    if min_val > 1e-4:
        print(f"  Smallest non-zero activation: {min_val:.6e}")
        print(f"  >> 1e-4, well above any noise threshold.")
        print(f"  CONCLUSION: SAE activations are STRICTLY SPARSE.")
        print(f"  Using '> 0' as activation criterion is CORRECT.")
    elif min_val > 1e-8:
        print(f"  Smallest non-zero activation: {min_val:.6e}")
        print(f"  Small but may be intentional. Inspect distribution above.")
    else:
        print(f"  Smallest non-zero activation: {min_val:.6e}")
        print(f"  Near-zero values exist. Consider adding a threshold.")


if __name__ == "__main__":
    main()
