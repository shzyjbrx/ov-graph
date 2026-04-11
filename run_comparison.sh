#!/bin/bash
#SBATCH --job-name=ov-viz
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/com_50.out
#SBATCH --error=logs/com_50.err

# ==============================================================================
# OV-Graph 预测对比实验启动脚本 (MIT-States)
# ==============================================================================

# 1. 创建日志目录
mkdir -p logs

# 2. 环境变量及 conda 激活
module purge
module load compilers/gcc/9.3.0
module load compilers/cuda/11.6
source ~/.bashrc
source activate /home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Base/miniconda3/envs/recAtk/xuan-czsl-py38

# 3. GPU 设置 (使用单卡进行推理)
export CUDA_VISIBLE_DEVICES=0

# 4. 运行主推理文件
echo "=========================================================="
echo "==> Starting Prediction Comparison: CLIP vs. OV-Graph..."
echo "==> Log is being written to logs/com_output.out"
echo "=========================================================="

python compare_predictions.py

echo "==> Experiment Finished."