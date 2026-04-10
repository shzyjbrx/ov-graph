#!/bin/bash
#SBATCH --job-name=cgqa-graph
#SBATCH --partition=gpu_mem
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/cgqa/train/Graph-%j.out
#SBATCH --error=logs/cgqa/train/Graph-%j.err

module purge
module load compilers/gcc/9.3.0
module load compilers/cuda/11.6
source ~/.bashrc
source activate /home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Base/miniconda3/envs/recAtk/xuan-czsl-py38

export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# [核心修改 1] 针对报错提示，限制 PyTorch 显存分配器的最大碎片大小，能极大缓解 40G 显卡的 OOM
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

DATA_ROOT="/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Base/data/cgqa"
LLM_DIR="/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/ov-graph/llm_nel_gen/cgqa_neighbors"
CONFIG="config/cgqa_softprompt_graph.yml"
N_GPU=4

echo "=========================================================="
echo "  Step3: 软提示 + RGCN 异构图注入 | C-GQA"
echo "  LLM 邻域目录: ${LLM_DIR}"
echo "  GPUs: ${N_GPU}"
echo "=========================================================="

python train.py --cfg ${CONFIG} \
    DATASET.root_dir           ${DATA_ROOT}                     \
    DISTRIBUTED.world_size     ${N_GPU}                         \
    TRAIN.checkpoint_dir       "checkpoints/cgqa/Graph/${SLURM_JOB_ID}" \
    TRAIN.log_dir              "tensorboards/cgqa/Graph/${SLURM_JOB_ID}" \
    TRAIN.batch_size           64                             \
    TRAIN.test_batch_size      256                              \
    TRAIN.sample_negative_pairs 512                             \
    TRAIN.lr                   5e-5                             \
    TRAIN.max_epoch            20                               \
    MODEL.n_ctx                16                               \
    MODEL.n_meta_ctx           4                                \
    MODEL.graph_dim            128                              \
    MODEL.graph_layers         2                                \
    MODEL.graph_dropout        0.4                              \
    MODEL.w_loss_attr          0.3                              \
    MODEL.w_loss_obj           0.3                              \
    MODEL.w_loss_neigh         0.1                              \
    MODEL.use_llm_nel          False                             \
    MODEL.llm_nel_dir          ${LLM_DIR}                       \
    MODEL.max_neighbors        2

echo "  训练完成"
echo "=========================================================="