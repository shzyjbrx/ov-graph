# OV-Graph

本项目实现了一种面向**开放词汇组合零样本学习（OV-CZSL）**的先进框架。针对模型在处理全新未见属性（Attribute）和物体（Object）组合时的泛化瓶颈，本项目通过大语言模型（LLM）构建异构语义常识图谱，并结合关系图卷积网络（RGCN）与大规模视觉语言预训练模型（CLIP）的提示学习（Soft Prompt Tuning），实现了图结构知识向多模态特征空间的完美对齐。

## ✨ 核心特性 (Key Features)

* **🧠 基于 LLM 的异构常识图谱 (`graph_module.py`)**
  利用 LLM 挖掘同义词与邻域词，构建包含 6 类节点（可见概念与邻域扩展概念）和 8 种拓扑关系边的异构图谱，为模型注入物理世界的常识先验。
* **🔗 图增强的残差软提示学习 (`clip_softprompt_graph.py`)**
  使用彻底冻结的 CLIP 作为骨干网络。通过 RGCN 在图谱上进行消息传递，利用 MetaNet 将图特征映射为动态 Token 并注入到独立软提示序列中。独创的残差锚定（Residual Anchoring）机制有效防止了灾难性遗忘。
* **🎯 多模态邻域对齐损失 (Neighborhood Smoothing Loss)**
  在文本特征空间中，利用纯文本 CLIP 先验相似度作为平滑权重，约束动态组装的软提示特征向 LLM 生成的合理组合邻域逼近，大幅提升模型对开放词汇的识别与重组能力。
* **⚡ 测试时双流特征融合 (Dual-stream Fusion)**
  在 Test-time 阶段，面对完全未见的概念，采用轻量级 Top-K 近邻图检索，并将原始文本特征与图谱迁移特征进行自适应加权融合（$\alpha e^{orig} + (1-\alpha)e^{graph}$），无需重新训练即可应对开放词汇挑战。
* **🚀 极致的工程优化**
  通过离线预计算图节点的 CLIP 零样本特征基准点，并在单卡 GPU 上实现秒级特征预热缓存与 DDP 分布式同步，彻底解决了庞大图谱加载导致的多卡通讯超时（Socket Timeout）问题。

---

## 📁 核心目录结构

```text
ov-graph/
├── config/                      # YAML 配置文件库
│   ├── defaults.py              # 默认超参数树
│   ├── mit_softprompt_graph.yml # MIT-States 模型配置文件
│   └── cgqa_softprompt_graph.yml# C-GQA 模型配置文件
├── models/                      # 网络架构核心实现
│   ├── clip_softprompt.py       # 基础残差软提示模型
│   ├── graph_module.py          # 异构图构建与 RGCN 传播模块
│   ├── clip_softprompt_graph.py # (Main) 图增强的软提示主模型
│   └── word_embedding_utils.py  # 词向量与文本预处理工具
├── llm_nel_gen/                 # 大语言模型生成的邻域词典
│   ├── mit-states_neighbors/    # MIT 属性/物体/组合 邻域 JSON
│   └── cgqa_neighbors/          # CGQA 邻域 JSON
├── train.py                     # DDP 分布式训练主脚本
├── dataset.py                   # CZSL 数据集与分割加载器
├── evaluator.py                 # AUC / HM 等指标评估器
├── train_mit_softprompt_graph.sh # MIT-States Slurm 启动脚本
└── train_cgqa_softprompt_graph.sh# C-GQA Slurm 启动脚本
```

---

## 🛠️ 环境依赖 (Installation)

本项目依赖 PyTorch、PyTorch Geometric (PyG) 以及 CLIP。您可以通过提供的环境文件直接恢复：

```bash
# 从 yaml 文件创建 Conda 环境
conda env create -f environment_ov_czsl.yml
conda activate xuan-czsl-py38

# 或者通过 pip 安装
pip install -r requirements.txt
```

---

## 📊 数据准备 (Data Preparation)

1. **图像数据集**：下载 `MIT-States` 和 `C-GQA` 数据集，并将其解压到指定的数据目录（如 `Base/data/`）。
2. **LLM 邻域字典**：确保 `llm_nel_gen/` 目录下包含通过 LLM 离线生成的近义词 `.json` 字典。字典格式如下：
    ```json
    {
      "ancient": ["historic", "old", "antient", "time-worn"]
    }
    ```

---

## 🚀 训练与评估 (Training & Evaluation)

本项目支持单卡及多卡 DDP (Distributed Data Parallel) 分布式训练。

### 在 MIT-States 上训练
我们在根目录下提供了标准的 SLURM 提交脚本，它已经配置好了完美的学习率、正则化权重以及图网络层数。

```bash
# 提交到 Slurm 集群（默认使用 2 张 GPU）
sbatch train_mit_softprompt_graph.sh
```

或者使用本地终端直接启动：
```bash
python train.py --cfg config/mit_softprompt_graph.yml \
    DATASET.root_dir /path/to/mit-states \
    MODEL.llm_nel_dir /path/to/mit_neighbors \
    DISTRIBUTED.world_size 2
```

### 在 C-GQA 上训练
C-GQA 数据集具有更大的规模和更细粒度的类别，我们采用了较小的 Batch Size (32) 和学习率 (5e-5)。

```bash
sbatch train_cgqa_softprompt_graph.sh
```

### 测试阶段
模型在训练过程中会自动在验证集（Validation）和测试集（Test）上进行评估，并自动保存具有最高 `AUC` 和 `Best HM` 指标的 `best_model.pth` 权重。所有的评估指标支持通过 TensorBoard 进行实时可视化。

---

## 📈 主要指标说明 (Metrics)

评估阶段会输出以下关键指标：
* **AUC (Area Under Curve)**：Seen accuracy 与 Unseen accuracy 构成的曲线下面积，是 CZSL 任务最核心的综合指标。
* **Best HM (Harmonic Mean)**：在不同偏置（Bias）下，Seen 和 Unseen 准确率的最佳调和平均数。
* **Unseen Acc**：模型对训练集中从未出现过的“属性-物体”组合的识别准确率（开放泛化能力的直接体现）。