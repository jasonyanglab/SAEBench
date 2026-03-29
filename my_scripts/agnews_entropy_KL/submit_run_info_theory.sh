#!/bin/bash
#SBATCH --job-name=info_theory
#SBATCH --account=project_2005865
#SBATCH --partition=gpusmall
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1

# 进入工作目录并激活已构建的 Tykky 环境与缓存重定向
cd /scratch/project_2005865/myj_SAE/project/SAEBench
source setup_env.sh

# 赋予执行权限并运行官方脚本
chmod +x my_scripts/agnews_entropy_KL/run_info_theory.sh
bash my_scripts/agnews_entropy_KL/run_info_theory.sh