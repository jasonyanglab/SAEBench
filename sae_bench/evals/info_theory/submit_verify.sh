#!/bin/bash
#SBATCH --job-name=verify_sae_features
#SBATCH --account=project_2005865
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1

# 进入工作目录并激活已构建的 Tykky 环境与缓存重定向
cd /scratch/project_2005865/myj_SAE/project/SAEBench
source setup_env.sh

# 运行验证脚本
python sae_bench/evals/info_theory/verify_class_features.py