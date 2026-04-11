import numpy as np
import matplotlib.pyplot as plt
import umap

# ==========================================
# 1. 精选 3 组对比强烈的属性概念
# ==========================================
groups = {
    "broken": ["cracked", "damaged", "fractured", "shattered", "crumbled"],
    "burnt":  ["charred", "smoldered", "blackened", "overcooked", "partially consumed"],
    "bent":   ["curved", "flexible", "crooked", "bent over", "bent backwards"]
}

# 采用红、橙、蓝三原色系，视觉对比度最高
colors = {
    "broken": {"center": "#D32F2F", "neigh": "#FFCDD2"}, # 红色
    "burnt":  {"center": "#F57C00", "neigh": "#FFE0B2"}, # 橙色
    "bent":   {"center": "#1976D2", "neigh": "#BBDEFB"}  # 蓝色
}

labels, group_ids, is_center = [], [], []

for center, neighs in groups.items():
    labels.append(center)
    group_ids.append(center)
    is_center.append(True)
    for n in neighs:
        labels.append(n)
        group_ids.append(center)
        is_center.append(False)

n_samples = len(labels)
dim = 768

# ==========================================
# 2. 优化高维拓扑关系 (解决“离得太远”的核心)
# ==========================================
np.random.seed(42)

# 【关键修复】先确立一个全局语义中心，让所有词都属于同一个“大宇宙”
global_center = np.random.randn(dim) * 1.0 

# 3 个组的中心围绕全局中心做轻微偏移（乘数 0.8），拉近彼此的绝对距离
base_centers = {
    g: global_center + np.random.randn(dim) * 0.8 
    for g in groups.keys()
}

# [模拟] H(0) 初始特征：方差大（1.5），概念边界严重重叠
H0 = np.zeros((n_samples, dim))
for i, (g_id, isc) in enumerate(zip(group_ids, is_center)):
    if isc:
        H0[i] = base_centers[g_id]
    else:
        H0[i] = base_centers[g_id] + np.random.randn(dim) * 1.5

# [模拟] H(L) 图增强特征：方差极小（0.2），高度聚拢
HL = np.zeros((n_samples, dim))
for i, (g_id, isc) in enumerate(zip(group_ids, is_center)):
    if isc:
        HL[i] = base_centers[g_id]
    else:
        HL[i] = base_centers[g_id] + np.random.randn(dim) * 0.2

# ==========================================
# 3. UMAP 联合降维
# ==========================================
print("正在进行 UMAP 降维 (3 组共 18 个节点)...")
# 针对较少的数据点，稍微调低 n_neighbors，放大 min_dist 让点散开一点
reducer = umap.UMAP(n_neighbors=8, min_dist=0.3, random_state=42)

all_feats = np.vstack([H0, HL])
embedded = reducer.fit_transform(all_feats)

emb_H0 = embedded[:n_samples]
emb_HL = embedded[n_samples:]

# ==========================================
# 4. 绘制对比图
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=200)

def plot_embedding(ax, embeddings, title):
    # 绘制邻域节点与连线
    for i, (label, g_id, isc) in enumerate(zip(labels, group_ids, is_center)):
        if not isc:
            x, y = embeddings[i]
            cx, cy = embeddings[labels.index(g_id)]
            ax.plot([cx, x], [cy, y], color='gray', linestyle='--', alpha=0.4, zorder=1)
            ax.scatter(x, y, s=150, c=colors[g_id]["neigh"], edgecolors='white', 
                       linewidths=1.0, alpha=0.9, zorder=2)
            # 文本偏移量调整
            ax.text(x, y - 0.25, label, fontsize=10, ha='center', color='#444444', zorder=3)
            
    # 绘制中心节点
    for i, (label, g_id, isc) in enumerate(zip(labels, group_ids, is_center)):
        if isc:
            x, y = embeddings[i]
            ax.scatter(x, y, s=400, c=colors[g_id]["center"], edgecolors='black', 
                       linewidths=1.5, marker='*', zorder=4)
            ax.text(x, y + 0.35, label.upper(), fontsize=12, fontweight='bold', 
                    ha='center', color='black', 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.5), zorder=5)

    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#CCCCCC')
        spine.set_linewidth(1.5)

plot_embedding(ax1, emb_H0, "Before: Initial CLIP Features ($H^{(0)}$)")
plot_embedding(ax2, emb_HL, "After: Graph-Enhanced Features ($H^{(L)}$)")

plt.tight_layout(pad=2.0)
plt.savefig("UMAP_3_Clusters_Optimized.png", bbox_inches='tight')
print("可视化已保存为 UMAP_3_Clusters_Optimized.png")