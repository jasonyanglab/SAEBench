#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --account=project_2005865
#SBATCH --partition=gputest
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=slurm_out/slurm-test_gpu-%j.out

# 强制 Tykky/Apptainer 容器绑定宿主机 GPU 驱动
export APPTAINER_NV=1

cd /scratch/project_2005865/myj_SAE/project/SAEBench
source my_scripts/setup_env.sh

# 运行两行极简 Python 代码测试 GPU 可见性
python -c "import torch; print('\n>>> CUDA Ready:', torch.cuda.is_available()); print('>>> GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None', '\n')"