#!/bin/bash
#SBATCH --job-name=info_theory_pii
#SBATCH --account=project_2005865
#SBATCH --partition=gpusmall
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=slurm_out/slurm-info_theory_pii-%j.out
#SBATCH --error=slurm_out/slurm-info_theory_pii-%j.err

cd /scratch/project_2005865/myj_SAE/project/SAEBench
source my_scripts/setup_env.sh

chmod +x my_scripts/pii_entropy_KL/run_info_theory_token.sh
bash my_scripts/pii_entropy_KL/run_info_theory_token.sh
