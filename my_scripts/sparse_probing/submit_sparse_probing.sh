#!/bin/bash
#SBATCH --job-name=sparse_probing
#SBATCH --account=project_2005865
#SBATCH --partition=gpusmall
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=slurm_out/slurm-sparse_probing-%j.out
#SBATCH --error=slurm_out/slurm-sparse_probing-%j.err

cd /scratch/project_2005865/myj_SAE/project/SAEBench
source my_scripts/setup_env.sh

chmod +x my_scripts/sparse_probing/run_sparse_probing.sh
bash my_scripts/sparse_probing/run_sparse_probing.sh
