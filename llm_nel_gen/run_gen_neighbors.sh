#!/bin/bash
#SBATCH --job-name=gen_neighbors         # 作业名称
#SBATCH --partition=gpu                  # 申请 GPU 分区
#SBATCH --gres=gpu:1                     # 申请 1 块 GPU (Qwen2.5-7B 建议使用 24GB 显存显卡)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00                  # 申请 12 小时
#SBATCH --output=logs/llm/mit-states-gen_neighbors_%j.out
#SBATCH --error=logs/llm/mit-states-gen_neighbors_%j.err

set -e

# 1. 加载系统模块
module purge
module load compilers/gcc/9.3.0
module load compilers/cuda/11.6

# 2. 激活 Conda 环境
# ⚠️ 请根据您的实际路径确认此环境路径
source ~/.bashrc
source activate /home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Base/miniconda3/envs/recAtk/xuan-czsl-py38

# 3. 网络与环境配置
export HF_ENDPOINT=https://hf-mirror.com
# export http_proxy=http://172.16.54.201:8888
# export https_proxy=http://172.16.54.201:8888
export PYTHONIOENCODING=utf-8

# 4. 核心参数配置
DATASET="mit-states"
# 对应您之前提到的 MIT-States 数据集根目录
DATA_ROOT="/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Base/data/mit-states"
# 推荐使用 Qwen2.5-7B-Instruct 获得更好的 JSON 遵循能力
MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
SAVE_DIR="/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/ov-czsl/llm_nel_gen/mit-states_neighbors"

# 确保日志和保存目录存在
mkdir -p logs/llm
mkdir -p ${SAVE_DIR}

echo "============================================================"
echo "  🚀 [LLM 启动] 正在为 OV-CZSL 生成结构化邻域词汇"
echo "  SLURM Job ID : ${SLURM_JOB_ID:-manual}"
echo "  数据集       : ${DATASET}"
echo "  模型         : ${MODEL_ID}"
echo "  输出目录     : ${SAVE_DIR}"
echo "  GPU 类型     : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "============================================================"

# 5. 执行生成脚本
# 注意：这里使用的是上一轮对话中提供的 generate_neighbors_en.py 脚本
python -u /home/bingxing2/home/scx6d4e/run/xuanzhenzhen/ov-czsl/llm_nel_gen/generate_neighborhoods.py \
    --data_root ${DATA_ROOT} \
    --save_dir ${SAVE_DIR} \
    --model_id ${MODEL_ID} \
    --temperature 0.2 \
    --max_new_tokens 512 \
    --max_retries 3

echo "============================================================"
echo "  ✅ [任务完成] 邻域文件已生成至: ${SAVE_DIR}"
echo "  包含文件: attr_neighbors.json, obj_neighbors.json, comp_neighbors.json"
echo "============================================================"