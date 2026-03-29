#!/bin/bash
#SBATCH --job-name=test_info_theory
#SBATCH --account=project_2005865
#SBATCH --partition=gputest
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G

# 1. 严格激活你的 Tykky 环境和环境变量
source setup_env.sh

# 2. 运行 10 条数据的极限测试
python sae_bench/evals/info_theory/main.py \
    --sae_regex_pattern "gemma-scope-2b-pt-res" \
    --sae_block_pattern ".*layer_12.*(16k).*" \
    --model_name "gemma-2-2b" \
    --llm_dtype "bfloat16" \
    --num_samples 10000 \
    --artifacts_path "artifacts/info_theory_test" \
    --output_folder "eval_results/info_theory_test" \
    --force_rerun