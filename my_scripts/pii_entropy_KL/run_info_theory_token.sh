#!/bin/bash

# User configuration
sae_regex_pattern="gemma-scope-2b-pt-res"
model_name="gemma-2-2b"
llm_dtype="bfloat16"

# Create array of patterns
declare -a sae_block_patterns=(
    ".*layer_5.*(16k).*"
    ".*layer_12.*(16k).*"
    ".*layer_19.*(16k).*"
)

for sae_block_pattern in "${sae_block_patterns[@]}"; do
    echo "Starting Token-level Info Theory eval for pattern ${sae_block_pattern}..."

    python sae_bench/evals/info_theory/main_token.py \
        --sae_regex_pattern "${sae_regex_pattern}" \
        --sae_block_pattern "${sae_block_pattern}" \
        --model_name ${model_name} \
        --llm_dtype ${llm_dtype} \
        --dataset_name "ai4privacy/pii-masking-300k" \
        --dataset_split "validation" \
        --num_samples 10000 \
        --include_non_entity \
        --force_rerun \
        --output_folder "eval_results/info_theory" || {
            echo "Token-level eval for pattern ${sae_block_pattern} failed, continuing..."
            continue
        }

    echo "Completed Token-level Info Theory eval for pattern ${sae_block_pattern}"
done
