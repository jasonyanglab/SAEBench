#!/bin/bash
#SBATCH --job-name=saebench_run_sh
#SBATCH --account=project_2005865
#SBATCH --partition=gpusmall
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1

#SBATCH --output=/scratch/project_2005865/myj_SAE/logs/slurm-%j.out
#SBATCH --error=/scratch/project_2005865/myj_SAE/logs/slurm-%j.err

# 进入工作目录并激活已构建的 Tykky 环境与缓存重定向
cd /projappl/project_2005865/myj_SAE/project/SAEBench
source setup_env.sh

# 赋予执行权限并运行官方脚本
chmod +x shell_scripts/run.sh
bash shell_scripts/run.sh