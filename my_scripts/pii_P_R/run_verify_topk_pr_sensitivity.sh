#!/bin/bash

# Sensitivity analysis: sweep max_h values
# Tests max_h in {0.3, 0.5, 0.7, 0.9} to find optimal P-R tradeoff
# Also includes the new KL+H group (H filter without density floor)

sae_regex_pattern="gemma-scope-2b-pt-res"
model_name="gemma-2-2b"
llm_dtype="bfloat16"

hkl_results_path="eval_results/info_theory/gemma-2-2b/pii-masking-300k_validation_n10000_ctx128_noO"

declare -a sae_block_patterns=(
    ".*layer_5.*(16k).*"
    ".*layer_12.*(16k).*"
    ".*layer_19.*(16k).*"
)

declare -a max_h_values=(0.5 0.6 0.7)

for max_h in "${max_h_values[@]}"; do
    output_folder="eval_results/topk_pr_verification_v8_span_maxh${max_h}"
    echo "============================================"
    echo "Running with max_h=${max_h}, output=${output_folder}"
    echo "============================================"

    for sae_block_pattern in "${sae_block_patterns[@]}"; do
        echo "  Pattern: ${sae_block_pattern}"

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
            --max_h ${max_h} \
            --drop_classes CARDISSUER \
            --hkl_results_path "${hkl_results_path}" \
            --output_folder "${output_folder}" \
            --artifacts_path "artifacts/info_theory" \
            --force_rerun || {
                echo "  Failed for ${sae_block_pattern}, continuing..."
                continue
            }

        echo "  Done: ${sae_block_pattern}"
    done

    echo "Finished max_h=${max_h}"
done

echo ""
echo "All sweeps done. Results in eval_results/topk_pr_verification_v6_span_maxh*/"
