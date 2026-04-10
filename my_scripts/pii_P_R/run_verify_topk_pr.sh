#!/bin/bash

# Top-k Feature P/R Verification for H/KL metrics
# Validates that KL-ranked features truly encode their assigned PII concepts
# by computing frequency-based Precision and Recall (noO mode, 25 entity types)
#
# Prerequisites: H/KL eval results must exist in hkl_results_path

sae_regex_pattern="gemma-scope-2b-pt-res"
model_name="gemma-2-2b"
llm_dtype="bfloat16"

# Path to existing H/KL evaluation results (noO)
hkl_results_path="eval_results/info_theory/gemma-2-2b/pii-masking-300k_validation_n10000_ctx128_noO"

declare -a sae_block_patterns=(
    ".*layer_5.*(16k).*"
    ".*layer_12.*(16k).*"
    ".*layer_19.*(16k).*"
)

for sae_block_pattern in "${sae_block_patterns[@]}"; do
    echo "Starting Top-k P/R verification for pattern ${sae_block_pattern}..."

    python sae_bench/evals/info_theory/verify_topk_features.py \
        --sae_regex_pattern "${sae_regex_pattern}" \
        --sae_block_pattern "${sae_block_pattern}" \
        --model_name ${model_name} \
        --llm_dtype ${llm_dtype} \
        --dataset_name "ai4privacy/pii-masking-300k" \
        --dataset_split "validation" \
        --num_samples 10000 \
        --k_values 1 5 10 20 \
        --n_random_trials 10 \
        --min_density 1e-3 \
        --max_density inf \
        --max_h 0.5 \
        --drop_classes CARDISSUER \
        --hkl_results_path "${hkl_results_path}" \
        --output_folder "eval_results/topk_pr_verification_v4_h" \
        --artifacts_path "artifacts/info_theory" \
        --force_rerun || {
            echo "Top-k P/R verification for pattern ${sae_block_pattern} failed, continuing..."
            continue
        }

    echo "Completed Top-k P/R verification for pattern ${sae_block_pattern}"
done
