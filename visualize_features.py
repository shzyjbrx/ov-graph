import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

# 导入您的项目组件
from config import cfg
from dataset import CompositionDataset
from models.clip_softprompt_graph import CLIPSoftPromptGraph

# ==========================================
# 1. 配置与模型加载 (参数强同步)
# ==========================================
CHECKPOINT_PATH = "/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/ov-graph/checkpoints/mit/Graph/1183025/mit_softprompt_graph_124/best_model.pth"
CONFIG_PATH = "config/mit_softprompt_graph.yml"

# 强制匹配参数以对齐 36108 条边的图结构
cfg.MODEL.n_ctx = 16
cfg.MODEL.use_llm_nel = True
cfg.MODEL.max_neighbors = 5
cfg.MODEL.llm_nel_dir = "/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/ov-czsl/llm_nel_gen/mit-states_neighbors"
cfg.DATASET.root_dir = "/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Base/data/mit-states"
cfg.DATASET.split_files_loc = "Data_files/MIT"

def load_ov_model():
    print(f"==> Loading Config and Dataset...")
    cfg.merge_from_file(CONFIG_PATH)
    
    # 【核心修复】：补全 dataset.py 解析路径所需的关键属性
    cfg.DATASET.dset_name = "mit"      # 用于拼接 "MIT_splits.pkl"
    cfg.DATASET.dset_split = 1        # 用于定位具体的划分版本
    cfg.DATASET.split_files_loc = "Data_files/MIT" 
    
    # 保持之前为了对齐权重而修正的参数
    cfg.MODEL.n_ctx = 16
    cfg.MODEL.use_llm_nel = True
    cfg.MODEL.max_neighbors = 5
    cfg.MODEL.llm_nel_dir = "/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/ov-czsl/llm_nel_gen/mit-states_neighbors"
    cfg.DATASET.root_dir = "/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Base/data/mit-states"
    
    # 强制单卡环境
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'

    # 初始化数据集
    trainset = CompositionDataset(phase='train', split=cfg.DATASET.splitname, cfg=cfg)
    
    print(f"==> Initializing OV-Graph Model...")
    model = CLIPSoftPromptGraph(trainset, cfg)
    
    print(f"==> Loading Weights from: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    return model

# ==========================================
# 2. 增强型特征提取逻辑 (自动识别节点身份)
# ==========================================
clusters_def = {
    'Ancient Group':  {'anchor': 'ancient',   'neighbors': ['historic', 'old', 'time-worn']},
    'Wet Group':      {'anchor': 'wet',       'neighbors': ['damp', 'moist', 'soggy']},
    'Broken Group':   {'anchor': 'broken',    'neighbors': ['cracked', 'damaged', 'shattered']},
    'Animal Group':   {'anchor': 'animal',    'neighbors': ['mammal', 'creature', 'dog']},
    'Vegetable Group':{'anchor': 'vegetable', 'neighbors': ['food', 'plant', 'fruit']}
}

@torch.no_grad()
def get_features_and_mapping(model):
    H0 = model.node_feats_init.cpu() 
    H_trained = model._propagate().cpu()
    gb = model.graph_builder
    results = {'orig': [], 'ours': [], 'labels': [], 'group': [], 'is_seen': []}
    neigh_to_anchor = {}

    def get_node_info(name):
        """核心修复：自动在 Seen 和 Neigh 字典中检索"""
        # 检索属性
        if name in gb.attr2id: return gb.attr2id[name], True
        if hasattr(gb, 'neigh_attr2id') and name in gb.neigh_attr2id: return gb.neigh_attr2id[name], False
        # 检索物体
        if name in gb.obj2id: return gb.obj2id[name], True
        if hasattr(gb, 'neigh_obj2id') and name in gb.neigh_obj2id: return gb.neigh_obj2id[name], False
        return None, False

    for g_name, nodes in clusters_def.items():
        anchor_word = nodes['anchor']
        a_idx, a_seen = get_node_info(anchor_word)
        
        if a_idx is not None:
            # 记录锚点
            results['orig'].append(H0[a_idx]); results['ours'].append(H_trained[a_idx])
            results['labels'].append(anchor_word); results['group'].append(g_name); results['is_seen'].append(a_seen)
            
            # 记录该锚点对应的邻居
            for n_word in nodes['neighbors']:
                n_idx, n_seen = get_node_info(n_word)
                if n_idx is not None:
                    results['orig'].append(H0[n_idx]); results['ours'].append(H_trained[n_idx])
                    results['labels'].append(n_word); results['group'].append(g_name); results['is_seen'].append(n_seen)
                    neigh_to_anchor[n_word] = anchor_word

    results['orig'] = torch.stack(results['orig']).numpy()
    results['ours'] = torch.stack(results['ours']).numpy()
    return results, neigh_to_anchor

# ==========================================
# 3. 绘图逻辑 (添加连接虚线)
# ==========================================
def plot_tsne_with_lines(res, mapping):
    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    all_feats = np.concatenate([res['orig'], res['ours']], axis=0)
    tsne = TSNE(n_components=2, perplexity=10, random_state=42)
    embs = tsne.fit_transform(all_feats)
    emb_list = [embs[:len(res['orig'])], embs[len(res['orig']):]]
    
    unique_groups = list(clusters_def.keys())
    palette = sns.color_palette("husl", len(unique_groups))
    g_colors = {g: palette[i] for i, g in enumerate(unique_groups)}

    for i, emb in enumerate(emb_list):
        ax = axes[i]
        label_to_xy = {l: xy for l, xy in zip(res['labels'], emb)}
        
        # 1. 绘点
        for g in unique_groups:
            for s_status in [True, False]:
                idxs = [j for j, (group, seen) in enumerate(zip(res['group'], res['is_seen'])) if group == g and seen == s_status]
                if idxs:
                    marker, size = ('*', 500) if s_status else ('o', 180)
                    ax.scatter(emb[idxs, 0], emb[idxs, 1], c=[g_colors[g]], marker=marker, s=size, edgecolors='w', label=f"{g} ({'Seen' if s_status else 'Neigh'})" if i==0 else "")
        
        # 2. 画虚线
        for neigh, anchor in mapping.items():
            if neigh in label_to_xy and anchor in label_to_xy:
                c = g_colors[res['group'][res['labels'].index(neigh)]]
                ax.plot([label_to_xy[anchor][0], label_to_xy[neigh][0]], [label_to_xy[anchor][1], label_to_xy[neigh][1]], color=c, ls='--', alpha=0.4, zorder=0)

        for j, txt in enumerate(res['labels']):
            ax.annotate(txt, (emb[j, 0], emb[j, 1]), fontsize=11, fontweight='bold', xytext=(5, 5), textcoords='offset points')

        ax.set_title(['(a) Original CLIP Space', '(b) OV-Graph Space'][i], fontsize=18, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])

    fig.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05), frameon=False)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig("tsne_final_clustering.png", dpi=300, bbox_inches='tight')
    print("✅ 可视化成功生成：tsne_final_clustering.png")

if __name__ == "__main__":
    model = load_ov_model()
    res, mapping = get_features_and_mapping(model)
    plot_tsne_with_lines(res, mapping)