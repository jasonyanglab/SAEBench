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
    echo "Starting Information Theory Alignment eval for pattern ${sae_block_pattern}..."

    python sae_bench/evals/info_theory/main.py \
        --sae_regex_pattern "${sae_regex_pattern}" \
        --sae_block_pattern "${sae_block_pattern}" \
        --model_name ${model_name} \
        --llm_dtype ${llm_dtype} \
        --dataset_name "fancyzhx/dbpedia_14" \
        --text_column "content" \
        --label_column "label" \
        --dataset_split "test" \
        --num_samples 10000 \
        --force_rerun \
        --output_folder "eval_results/info_theory" || {
            echo "Info Theory eval for pattern ${sae_block_pattern} failed, continuing to next pattern..."
            continue
        }

    echo "Completed Information Theory Alignment eval for pattern ${sae_block_pattern}"
done
