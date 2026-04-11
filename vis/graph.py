import networkx as nx
import matplotlib.pyplot as plt

# ==========================================
# 1. 定义图结构与节点数据
# ==========================================
# 使用有向图，以支持单向和双向（通过两条反向边+弧度）箭头
G = nx.DiGraph()

# 定义节点及其属性：坐标位置(pos) 和 颜色(color)
# 颜色参考：Seen(深色), Neigh(浅色) -> 绿(组合), 蓝(属性), 橙(物体)
nodes_data = {
    # 组合 (Composition)
    'sliced apple': {'pos': (0, 2),    'color': '#2ca02c', 'label': 'sliced apple\n(C_seen)'},
    'peeled apple': {'pos': (-2.5, 2), 'color': '#98df8a', 'label': 'peeled apple\n(C_neigh)'},
    'sliced pear':  {'pos': (2.5, 2),  'color': '#98df8a', 'label': 'sliced pear\n(C_neigh)'},
    
    # 属性 (Attribute)
    'sliced':       {'pos': (-1, 0),   'color': '#1f77b4', 'label': 'sliced\n(A_seen)'},
    'cut':          {'pos': (-2.5, 0.5), 'color': '#aec7e8', 'label': 'cut\n(A_neigh)'},
    'chopped':      {'pos': (-2.5, -0.5),'color': '#aec7e8', 'label': 'chopped\n(A_neigh)'},
    
    # 物体 (Object)
    'apple':        {'pos': (1, 0),    'color': '#ff7f0e', 'label': 'apple\n(O_seen)'},
    'pear':         {'pos': (2.5, 0.5),'color': '#ffbb78', 'label': 'pear\n(O_neigh)'},
    'peach':        {'pos': (2.5, -0.5),'color': '#ffbb78', 'label': 'peach\n(O_neigh)'},
}

for node, data in nodes_data.items():
    G.add_node(node, pos=data['pos'], color=data['color'], label=data['label'])

pos = nx.get_node_attributes(G, 'pos')
colors = [nx.get_node_attributes(G, 'color')[node] for node in G.nodes()]
labels = nx.get_node_attributes(G, 'label')

# ==========================================
# 2. 定义边 (分类存放以便应用不同的绘图样式)
# ==========================================
# (1) 组合构成边 (R_attr, R_obj)：单向黑实线
edges_compose = [
    ('sliced', 'sliced apple'),
    ('apple', 'sliced apple')
]

# (2) 共现关系边 (R_co)：双向橙黄实线 (A <-> O)
edges_co_occur = [
    ('sliced', 'apple'), 
    ('apple', 'sliced')
]

# (3) 语义扩展边 (R_sem)：双向红虚线 (Seen <-> Neigh)
edges_semantic_pairs = [
    ('sliced apple', 'peeled apple'),
    ('sliced apple', 'sliced pear'),
    ('sliced', 'cut'),
    ('sliced', 'chopped'),
    ('apple', 'pear'),
    ('apple', 'peach')
]
edges_semantic = []
for u, v in edges_semantic_pairs:
    edges_semantic.extend([(u, v), (v, u)])  # 添加双向边

G.add_edges_from(edges_compose + edges_co_occur + edges_semantic)

# ==========================================
# 3. 开始可视化绘制
# ==========================================
fig, ax = plt.subplots(figsize=(10, 7), dpi=150)

# 绘制节点
nx.draw_networkx_nodes(
    G, pos, ax=ax,
    node_color=colors,
    node_size=4000, 
    edgecolors='white', # 节点白边
    linewidths=2
)

# 绘制节点文本
nx.draw_networkx_labels(
    G, pos, labels=labels, ax=ax,
    font_size=9, font_weight='bold', font_color='black'
)

# 绘制组合构成边 (单向，黑实线)
nx.draw_networkx_edges(
    G, pos, ax=ax, edgelist=edges_compose,
    edge_color='black', width=2, arrowsize=20,
    connectionstyle='arc3,rad=0.05' # 轻微弧度防止死板
)

# 绘制共现关系边 (双向，橙黄实线)
nx.draw_networkx_edges(
    G, pos, ax=ax, edgelist=edges_co_occur,
    edge_color='#ffbc00', width=2.5, arrowsize=20,
    connectionstyle='arc3,rad=0.1' # 弧度让双向箭头不重叠
)

# 绘制语义扩展边 (双向，红虚线)
nx.draw_networkx_edges(
    G, pos, ax=ax, edgelist=edges_semantic,
    edge_color='#d62728', width=1.5, style='dashed', arrowsize=15,
    connectionstyle='arc3,rad=0.1' 
)

# ==========================================
# 4. 图例与排版优化
# ==========================================
# 手动添加自定义图例
import matplotlib.lines as mlines
legend_elements = [
    mlines.Line2D([], [], color='black', marker='>', markersize=8, label=r'Attr/Obj $\rightarrow$ Comp', lw=2),
    mlines.Line2D([], [], color='#ffbc00', marker='>', markersize=8, label=r'Co-occurrence (Attr $\leftrightarrow$ Obj)', lw=2.5),
    mlines.Line2D([], [], color='#d62728', marker='>', markersize=8, label=r'Semantic Expand (Seen $\leftrightarrow$ Neigh)', lw=1.5, linestyle='--')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10, frameon=True, shadow=True)

ax.set_title("Heterogeneous Concept Graph (OV-CZSL)", fontsize=16, fontweight='bold', pad=20)
ax.axis('off') # 隐藏坐标轴
plt.tight_layout()

# 保存或展示
plt.savefig("heterogeneous_graph_reproduced.png", bbox_inches='tight')
plt.show()