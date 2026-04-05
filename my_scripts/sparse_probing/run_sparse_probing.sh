#!/bin/bash

# Sparse probing evaluation on the same SAEs as info_theory experiments
# Default config runs on 8 datasets (including ag_news for direct comparison)
# Context length 128, same as info_theory

sae_regex_pattern="gemma-scope-2b-pt-res"
model_name="gemma-2-2b"
llm_dtype="bfloat16"

declare -a sae_block_patterns=(
    ".*layer_5.*(16k).*"
    ".*layer_12.*(16k).*"
    ".*layer_19.*(16k).*"
)

for sae_block_pattern in "${sae_block_patterns[@]}"; do
    echo "Starting Sparse Probing eval for pattern ${sae_block_pattern}..."

    python -m sae_bench.evals.sparse_probing.main \
        --sae_regex_pattern "${sae_regex_pattern}" \
        --sae_block_pattern "${sae_block_pattern}" \
        --model_name ${model_name} \
        --llm_dtype ${llm_dtype} \
        --dataset_names "fancyzhx/ag_news" \
        --force_rerun \
        --output_folder "eval_results/sparse_probing" \
        --artifacts_path "artifacts/sparse_probing" || {
            echo "Sparse Probing eval for pattern ${sae_block_pattern} failed, continuing..."
            continue
        }

    echo "Completed Sparse Probing eval for pattern ${sae_block_pattern}"
done
