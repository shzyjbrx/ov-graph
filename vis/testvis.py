import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.patches as mpatches

# ==========================================
# 1. 设置学术论文级绘图参数
# ==========================================
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")

COLOR_SEEN = '#2b5c8f'    
COLOR_NEIGH = '#e76f51'   

# ==========================================
# 2. 模拟双流特征融合的 Top-K 检索数据
# ==========================================
data = {
    'rusty car': {
        'Attr (rusty)':  [('weathered', 'Seen', 0.40), ('corroded', 'Neigh', 0.35), ('oxidized', 'Neigh', 0.15), ('browned', 'Seen', 0.10)],
        'Obj (car)':     [('bus', 'Seen', 0.45), ('automobile', 'Neigh', 0.30), ('vehicle', 'Neigh', 0.15), ('truck', 'Neigh', 0.10)],
        'Comp (rusty car)':[('broken bus', 'Seen', 0.50), ('weathered vehicle', 'Neigh', 0.30), ('corroded truck', 'Neigh', 0.20)]
    },
    'thawed fish': {
        'Attr (thawed)': [('wet', 'Seen', 0.40), ('melted', 'Neigh', 0.30), ('unfrozen', 'Neigh', 0.20), ('warm', 'Neigh', 0.10)],
        'Obj (fish)':    [('seafood', 'Neigh', 0.45), ('salmon', 'Neigh', 0.30), ('meat', 'Seen', 0.15), ('animal', 'Neigh', 0.10)],
        'Comp (thawed fish)':[('wet meat', 'Seen', 0.45), ('defrosted seafood', 'Neigh', 0.35), ('cooked salmon', 'Neigh', 0.20)]
    },
    'pureed vegetable': {
        'Attr (pureed)':[('whipped', 'Seen', 0.40), ('mashed', 'Neigh', 0.35), ('blended', 'Neigh', 0.15), ('crushed', 'Neigh', 0.10)],
        'Obj (vegetable)':[('plant', 'Seen', 0.40), ('food', 'Neigh', 0.30), ('produce', 'Neigh', 0.20), ('crop', 'Neigh', 0.10)],
        'Comp (pureed veg)':[('whipped salad', 'Seen', 0.50), ('mashed food', 'Neigh', 0.30), ('blended plant', 'Neigh', 0.20)]
    }
}

# ==========================================
# 3. 绘制 3x3 核心图表
# ==========================================
fig, axes = plt.subplots(3, 3, figsize=(16, 10))

# 缩小标题字体
fig.suptitle('Test-time Dual-stream Fusion: Top-K Neighborhood Attention Weights', 
             fontsize=15, fontweight='bold', y=0.98)

targets = list(data.keys())

for row_idx, target in enumerate(targets):
    branches = list(data[target].keys())
    for col_idx, branch in enumerate(branches):
        ax = axes[row_idx, col_idx]
        
        nodes = data[target][branch][::-1]
        names = [f"[{n[1]}] {n[0]}" for n in nodes]
        weights = [n[2] for n in nodes]
        colors = [COLOR_SEEN if n[1] == 'Seen' else COLOR_NEIGH for n in nodes]
        
        bars = ax.barh(names, weights, color=colors, edgecolor='none', height=0.6)
        
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}', va='center', fontsize=10, color='black')

        ax.set_xlim(0, 0.6)
        
        # 仅在第一行显示分支类型标题
        if row_idx == 0:
            ax.set_title(f"Branch: {branch.split(' ')[0]}", fontsize=13, fontweight='bold', pad=10)
            
        # 【新增】：在最左侧列添加测试用例标签 (Test Case)
        if col_idx == 0:
            ax.set_ylabel(f"Test Case:\n'{target}'", fontsize=14, fontweight='bold', 
                          rotation=0, ha='right', va='center', labelpad=20, color='#333333')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', labelsize=11)
        ax.set_xlabel("Attention Weight" if row_idx == 2 else "", fontsize=10)

seen_patch = mpatches.Patch(color=COLOR_SEEN, label='Seen Node (Training Set)')
neigh_patch = mpatches.Patch(color=COLOR_NEIGH, label='Neighborhood Node (LLM Prior)')
fig.legend(handles=[seen_patch, neigh_patch], loc='lower center', ncol=2, 
           fontsize=12, bbox_to_anchor=(0.5, 0.02), frameon=False)

# 调整左侧边距以留出空间给 ylabel
plt.tight_layout(rect=[0.05, 0.05, 1, 0.95]) 

# ==========================================
# 4. 输出并保存
# ==========================================
plt.savefig("dual_stream_fusion_attention_labeled.png", dpi=600, bbox_inches='tight')
plt.show()