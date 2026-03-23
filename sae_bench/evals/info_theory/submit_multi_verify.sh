#!/bin/bash
#SBATCH --job-name=multi_layer_verify
#SBATCH --account=project_2005865
#SBATCH --partition=gpusmall
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=slurm-multi-verify-%j.out
#SBATCH --error=slurm-multi-verify-%j.err

cd /scratch/project_2005865/myj_SAE/project/SAEBench
source setup_env.sh
python sae_bench/evals/info_theory/run_multi_layer_verify.py