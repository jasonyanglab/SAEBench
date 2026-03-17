#!/bin/bash

# ==========================================
# 1. 目录存在性校验与自动创建
# ==========================================
# 确保所有声明的输出和缓存目录均已物理存在，避免运行时报错
mkdir -p /scratch/project_2005865/myj_SAE/hf_cache/transformers
mkdir -p /scratch/project_2005865/myj_SAE/hf_cache/datasets
mkdir -p /scratch/project_2005865/myj_SAE/tmp
mkdir -p /scratch/project_2005865/myj_SAE/outputs
mkdir -p /scratch/project_2005865/myj_SAE/logs

# ==========================================
# 2. 核心缓存与临时文件目录设置 (避开 Home 目录限制)
# ==========================================
export HF_HOME=/scratch/project_2005865/myj_SAE/hf_cache
export TRANSFORMERS_CACHE=/scratch/project_2005865/myj_SAE/hf_cache/transformers
export HF_DATASETS_CACHE=/scratch/project_2005865/myj_SAE/hf_cache/datasets

export TMPDIR=/scratch/project_2005865/myj_SAE/tmp
export APPTAINER_TMPDIR=$TMPDIR

# ==========================================
# 3. 项目结构与输出路径全局变量
# ==========================================
export PROJECT_ROOT=/scratch/project_2005865/myj_SAE
export SCRATCH_ROOT=/scratch/project_2005865/myj_SAE
export OUTPUT_ROOT=/scratch/project_2005865/myj_SAE/outputs
export LOG_ROOT=/scratch/project_2005865/myj_SAE/logs

# ==========================================
# 4. Python 行为控制与环境激活
# ==========================================
export PYTHONUNBUFFERED=1
export PATH="/scratch/project_2005865/myj_SAE/project/SAEBench/saebench_env/bin:$PATH"

# 强迫 Python 优先读取当前目录的源码，防止调用只读容器内的包
export PYTHONPATH="/scratch/project_2005865/myj_SAE/project/SAEBench:$PYTHONPATH"
# 强制 Tykky/Apptainer 容器绑定宿主机 GPU 驱动
export APPTAINER_NV=1

echo "Mahti 超算 SAEBench 运行环境已就绪。"