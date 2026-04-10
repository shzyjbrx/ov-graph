
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# pip install umap-learn
import umap

# =========================================================
# 1. Hard-coded example data
#    This script does NOT read your json/txt files.
#    It directly uses a coherent toy semantic graph.
# =========================================================
BASE = Path("./vis")

seen_attrs = ["fresh", "burnt", "caramelized"]
seen_objs = ["bread", "banana", "cheese", "coffee", "lemon"]
seen_comps = [
    "fresh bread",
    "fresh coffee",
    "fresh lemon",
    "burnt bread",
    "caramelized banana",
    "caramelized bread",
    "caramelized cheese",
    "caramelized coffee",
    "caramelized lemon",
]

attr_neigh = {
    "fresh": ["new", "crisp"],
    "burnt": ["charred", "blackened"],
    "caramelized": ["golden brown", "sugary toasted"],
}
obj_neigh = {
    "bread": ["cake"],
    "banana": ["apple"],
    "cheese": ["milk"],
    "coffee": ["tea"],
    "lemon": ["orange"],
}
comp_neigh = {
    "fresh bread": ["fresh baked bread"],
    "fresh coffee": ["newly brewed coffee"],
    "fresh lemon": ["juicy lemon"],
    "burnt bread": ["charred bread"],
    "caramelized banana": ["golden banana"],
    "caramelized bread": ["toasted bread"],
    "caramelized cheese": ["golden cheese"],
    "caramelized coffee": ["richly roasted coffee"],
    "caramelized lemon": ["sweetened lemon"],
}


# =========================================================
# 2. Build node list
# =========================================================
nodes = []

def add_node(label, node_type, domain, group_key):
    nodes.append(
        {
            "label": label,       # node text
            "type": node_type,    # attr / obj / comp
            "domain": domain,     # seen / neigh
            "group": group_key,   # semantic family id
        }
    )

for a in seen_attrs:
    add_node(a, "attr", "seen", a)
    for n in attr_neigh[a]:
        add_node(n, "attr", "neigh", a)

for o in seen_objs:
    add_node(o, "obj", "seen", o)
    for n in obj_neigh[o]:
        add_node(n, "obj", "neigh", o)

for c in seen_comps:
    add_node(c, "comp", "seen", c)
    for n in comp_neigh.get(c, []):
        add_node(n, "comp", "neigh", c)


# =========================================================
# 3. Create idealized "before / after RGCN" embeddings
# =========================================================
rng = np.random.default_rng(42)

# rough CLIP-initialized layout
attr_centers_before = {
    "fresh": np.array([-3.3,  2.0]),
    "burnt": np.array([-3.2,  0.0]),
    "caramelized": np.array([-3.1, -2.0]),
}
obj_centers_before = {
    "bread": np.array([ 3.1,  2.2]),
    "banana": np.array([ 3.5,  1.0]),
    "cheese": np.array([ 3.2,  0.0]),
    "coffee": np.array([ 3.5, -1.0]),
    "lemon": np.array([ 3.1, -2.1]),
}
comp_centers_before = {
    "fresh bread": np.array([ 0.1,  1.8]),
    "fresh coffee": np.array([ 0.2,  0.3]),
    "fresh lemon": np.array([-0.1, -0.7]),
    "burnt bread": np.array([-0.3,  0.9]),
    "caramelized banana": np.array([ 0.6,  0.2]),
    "caramelized bread": np.array([ 0.1,  0.7]),
    "caramelized cheese": np.array([ 0.2, -0.4]),
    "caramelized coffee": np.array([ 0.4, -1.1]),
    "caramelized lemon": np.array([ 0.0, -1.6]),
}

# graph-enhanced layout
attr_centers_after = {
    "fresh": np.array([-2.7,  1.8]),
    "burnt": np.array([-2.8,  0.2]),
    "caramelized": np.array([-2.7, -1.5]),
}
obj_centers_after = {
    "bread": np.array([ 2.5,  2.0]),
    "banana": np.array([ 2.9,  1.0]),
    "cheese": np.array([ 2.7,  0.0]),
    "coffee": np.array([ 2.9, -1.0]),
    "lemon": np.array([ 2.5, -1.9]),
}
comp_centers_after = {
    "fresh bread": np.array([-0.2,  1.9]),
    "fresh coffee": np.array([-0.1,  0.5]),
    "fresh lemon": np.array([-0.2, -0.2]),
    "burnt bread": np.array([-0.4,  0.9]),
    "caramelized banana": np.array([ 0.2, -0.2]),
    "caramelized bread": np.array([-0.1,  0.3]),
    "caramelized cheese": np.array([ 0.1, -0.7]),
    "caramelized coffee": np.array([ 0.3, -1.1]),
    "caramelized lemon": np.array([ 0.0, -1.3]),
}

def sample_point(center, domain, after=False):
    if after:
        scale = 0.10 if domain == "seen" else 0.16
    else:
        scale = 0.20 if domain == "seen" else 0.32
    return center + rng.normal(0, scale, size=2)

H0, HL = [], []
for node in nodes:
    if node["type"] == "attr":
        c0, c1 = attr_centers_before[node["group"]], attr_centers_after[node["group"]]
    elif node["type"] == "obj":
        c0, c1 = obj_centers_before[node["group"]], obj_centers_after[node["group"]]
    else:
        c0, c1 = comp_centers_before[node["group"]], comp_centers_after[node["group"]]

    H0.append(sample_point(c0, node["domain"], after=False))
    HL.append(sample_point(c1, node["domain"], after=True))

H0 = np.stack(H0)
HL = np.stack(HL)

# Joint UMAP for a paper-style layout
all_feat = np.concatenate([H0, HL], axis=0)
reducer = umap.UMAP(
    n_neighbors=10,
    min_dist=0.35,
    metric="euclidean",
    random_state=42,
)
all_2d = reducer.fit_transform(all_feat)
H0_2d = all_2d[:len(nodes)]
HL_2d = all_2d[len(nodes):]


# =========================================================
# 4. Plot
# =========================================================
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

color_map = {
    "attr": "#E68613",   # orange
    "obj": "#4DAF4A",    # green
    "comp": "#377EB8",   # blue
}
marker_map = {
    "seen": "o",
    "neigh": "^",
}
type_name = {
    "attr": "Attribute",
    "obj": "Object",
    "comp": "Composition",
}
domain_name = {
    "seen": "Seen",
    "neigh": "Neighbor",
}

fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.0), dpi=220)

def draw_panel(ax, feat, title):
    for node_type in ["attr", "obj", "comp"]:
        idx_all = [i for i, n in enumerate(nodes) if n["type"] == node_type]
        ax.scatter(
            feat[idx_all, 0],
            feat[idx_all, 1],
            s=95,
            c=color_map[node_type],
            alpha=0.08,
            linewidths=0,
            zorder=1,
        )

        for domain in ["seen", "neigh"]:
            idx = [i for i, n in enumerate(nodes)
                   if n["type"] == node_type and n["domain"] == domain]
            if not idx:
                continue

            if domain == "seen":
                ax.scatter(
                    feat[idx, 0], feat[idx, 1],
                    s=52,
                    c=color_map[node_type],
                    marker=marker_map[domain],
                    edgecolors="white",
                    linewidths=0.8,
                    alpha=0.96,
                    label=f"{domain_name[domain]}-{type_name[node_type]}",
                    zorder=3,
                )
            else:
                ax.scatter(
                    feat[idx, 0], feat[idx, 1],
                    s=58,
                    facecolors="white",
                    edgecolors=color_map[node_type],
                    marker=marker_map[domain],
                    linewidths=1.1,
                    alpha=0.95,
                    label=f"{domain_name[domain]}-{type_name[node_type]}",
                    zorder=2,
                )

    highlight_labels = {
        "fresh", "burnt", "caramelized",
        "bread", "banana", "cheese", "coffee", "lemon",
        "fresh bread", "burnt bread", "caramelized banana",
        "charred", "golden brown", "cake", "tea", "orange",
        "fresh baked bread", "golden banana", "richly roasted coffee"
    }

    for i, node in enumerate(nodes):
        if node["label"] in highlight_labels:
            x, y = feat[i]
            ax.text(
                x + 0.06, y + 0.04,
                node["label"],
                fontsize=8.2,
                color="#222222",
                zorder=5,
            )

    ax.set_title(title, pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

draw_panel(axes[0], H0_2d, "Before RGCN")
draw_panel(axes[1], HL_2d, "After RGCN")

handles, labels = axes[1].get_legend_handles_labels()
unique = {}
for h, l in zip(handles, labels):
    if l not in unique:
        unique[l] = h

fig.legend(
    unique.values(),
    unique.keys(),
    loc="lower center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, -0.01),
    columnspacing=1.8,
    handletextpad=0.6,
)

fig.suptitle(
    "Idealized node distribution before and after RGCN propagation",
    y=1.02,
    fontsize=15,
)

plt.tight_layout(rect=[0, 0.05, 1, 0.98])

out_png = BASE / "ideal_rgcn_before_after_paper_style_hardcoded.png"
out_pdf = BASE / "ideal_rgcn_before_after_paper_style_hardcoded.pdf"
plt.savefig(out_png, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
plt.close()

print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
