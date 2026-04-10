"""
graph_module.py  —— OV-CZSL 异构图构建器 + RGCN 传播模块

节点类型 (6类):
    Aseen, Oseen, Cseen   : 训练集已见概念
    Aneigh, Oneigh, Cneigh : LLM 扩展的邻域概念

边类型 (8类):
    R0  ATTR_IN_COMP  A → C         组合构成（属性方向）
    R1  COMP_HAS_ATTR C → A
    R2  OBJ_IN_COMP   O → C         组合构成（物体方向）
    R3  COMP_HAS_OBJ  C → O
    R4  ATTR_CO_OBJ   A → O         训练集共现
    R5  OBJ_CO_ATTR   O → A
    R6  COMP_SIBLING  C → C         共享属性/物体的组合相邻
    R7  SEM_EXPAND    Seen ↔ Neigh  语义扩展（双向）
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class HeteroGraphBuilder:
    NUM_RELATIONS = 8
    REL_NAMES = [
        'ATTR_IN_COMP', 'COMP_HAS_ATTR',
        'OBJ_IN_COMP',  'COMP_HAS_OBJ',
        'ATTR_CO_OBJ',  'OBJ_CO_ATTR',
        'COMP_SIBLING', 'SEM_EXPAND',
    ]

    def __init__(self, dset, neighbor_paths=None, max_neighbors=5):
        """
        Args:
            dset            : CompositionDataset
            neighbor_paths  : dict {'attr': path, 'obj': path, 'comp': path}
                              None = 仅使用 Seen 节点（无 LLM 扩展）
            max_neighbors   : 每个 Seen 节点最多保留的邻域节点数
        """
        self.dset    = dset
        self.max_k   = max_neighbors

        # 邻域词汇存储
        self.attr_neighbors = {}   # attr_name  -> [word, ...]
        self.obj_neighbors  = {}   # obj_name   -> [word, ...]
        self.comp_neighbors = {}   # (attr, obj)-> [phrase, ...]

        if neighbor_paths is not None:
            self._load_neighbors(neighbor_paths)

        self._build_nodes()
        self._build_edges()

    # ─────────────────────────────────────────────────────────
    # 数据加载
    # ─────────────────────────────────────────────────────────
    def _load_neighbors(self, paths):
        def _load(path):
            if path and path != '' :
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except Exception as e:
                    print(f"[Graph] 警告：无法加载 {path} : {e}")
            return {}

        raw_attr = _load(paths.get('attr', ''))
        raw_obj  = _load(paths.get('obj',  ''))
        raw_comp = _load(paths.get('comp', ''))

        for k, v in raw_attr.items():
            self.attr_neighbors[k.lower()] = [w.strip().lower() for w in v[:self.max_k]]

        for k, v in raw_obj.items():
            self.obj_neighbors[k.lower()] = [w.strip().lower() for w in v[:self.max_k]]

        for k, v in raw_comp.items():
            # key 格式: "attr obj" 或 "attr_obj"
            parts = k.strip().split()
            if len(parts) >= 2:
                key = (parts[0].lower(), ' '.join(parts[1:]).lower())
            else:
                key = (k.lower(), '')
            self.comp_neighbors[key] = [p.strip().lower() for p in v[:self.max_k]]

        print(f"[Graph] LLM 邻域加载: "
              f"attr={len(self.attr_neighbors)}, "
              f"obj={len(self.obj_neighbors)}, "
              f"comp={len(self.comp_neighbors)}")

    # ─────────────────────────────────────────────────────────
    # 节点构建
    # ─────────────────────────────────────────────────────────
    def _build_nodes(self):
        d = self.dset

        # ── Seen 节点 ──────────────────────────────────────
        self.seen_attrs = list(d.train_attrs)
        self.seen_objs  = list(d.train_objs)
        self.seen_comps = list(d.train_pairs)   # [(attr, obj), ...]

        nA = len(self.seen_attrs)
        nO = len(self.seen_objs)
        nC = len(self.seen_comps)

        self.attr_offset = 0
        self.obj_offset  = nA
        self.comp_offset = nA + nO

        self.attr2id = {a: i          for i, a in enumerate(self.seen_attrs)}
        self.obj2id  = {o: i + nA     for i, o in enumerate(self.seen_objs)}
        self.comp2id = {c: i + nA + nO for i, c in enumerate(self.seen_comps)}

        # ── Neigh 节点（去重 + 过滤已有 Seen 概念）──────────
        seen_attr_set  = set(self.seen_attrs)
        seen_obj_set   = set(self.seen_objs)
        seen_comp_set  = {f"{a} {o}" for a, o in self.seen_comps}

        neigh_attr_set = set()
        neigh_obj_set  = set()
        neigh_comp_set = set()

        for words in self.attr_neighbors.values():
            for w in words:
                if w and w not in seen_attr_set:
                    neigh_attr_set.add(w)

        for words in self.obj_neighbors.values():
            for w in words:
                if w and w not in seen_obj_set:
                    neigh_obj_set.add(w)

        for phrases in self.comp_neighbors.values():
            for ph in phrases:
                if ph and ph not in seen_comp_set:
                    neigh_comp_set.add(ph)

        self.neigh_attrs = sorted(neigh_attr_set)
        self.neigh_objs  = sorted(neigh_obj_set)
        self.neigh_comps = sorted(neigh_comp_set)

        nAN = len(self.neigh_attrs)
        nON = len(self.neigh_objs)
        nCN = len(self.neigh_comps)

        base = nA + nO + nC
        self.neigh_attr_offset = base
        self.neigh_obj_offset  = base + nAN
        self.neigh_comp_offset = base + nAN + nON

        self.neigh_attr2id = {a: i + base           for i, a in enumerate(self.neigh_attrs)}
        self.neigh_obj2id  = {o: i + base + nAN     for i, o in enumerate(self.neigh_objs)}
        self.neigh_comp2id = {c: i + base + nAN + nON for i, c in enumerate(self.neigh_comps)}

        self.num_nodes = nA + nO + nC + nAN + nON + nCN

        print(f"[Graph] 节点统计: "
              f"Aseen={nA}, Oseen={nO}, Cseen={nC}, "
              f"Aneigh={nAN}, Oneigh={nON}, Cneigh={nCN}, "
              f"Total={self.num_nodes}")

    # ─────────────────────────────────────────────────────────
    # 边构建
    # ─────────────────────────────────────────────────────────
    def _build_edges(self):
        edges = {r: [] for r in range(self.NUM_RELATIONS)}

        # R0/R1/R2/R3: 组合构成关系（Seen 内部）
        for (a, o) in self.seen_comps:
            if a not in self.attr2id or o not in self.obj2id:
                continue
            ai = self.attr2id[a]
            oi = self.obj2id[o]
            ci = self.comp2id[(a, o)]
            edges[0].append((ai, ci))   # A → C
            edges[1].append((ci, ai))   # C → A
            edges[2].append((oi, ci))   # O → C
            edges[3].append((ci, oi))   # C → O

        # R4/R5: 属性-物体共现（训练集真实搭配）
        co_pairs = set()
        for _, attr, obj in self.dset.train_data:
            if attr in self.attr2id and obj in self.obj2id:
                co_pairs.add((attr, obj))
        for (a, o) in co_pairs:
            edges[4].append((self.attr2id[a], self.obj2id[o]))   # A → O
            edges[5].append((self.obj2id[o],  self.attr2id[a]))  # O → A

        # R6: 组合相邻关系（共享属性或物体）
        attr_to_comps = {}
        obj_to_comps  = {}
        for (a, o) in self.seen_comps:
            attr_to_comps.setdefault(a, []).append((a, o))
            obj_to_comps.setdefault(o, []).append((a, o))

        sibling_set = set()
        for group in list(attr_to_comps.values()) + list(obj_to_comps.values()):
            for c1 in group:
                for c2 in group:
                    if c1 != c2:
                        e = (self.comp2id[c1], self.comp2id[c2])
                        if e not in sibling_set:
                            sibling_set.add(e)
                            edges[6].append(e)

        # R7: 语义扩展（Seen ↔ Neigh，双向）
        for a, words in self.attr_neighbors.items():
            if a not in self.attr2id:
                continue
            ai = self.attr2id[a]
            for w in words:
                if w in self.neigh_attr2id:
                    ni = self.neigh_attr2id[w]
                    edges[7].append((ai, ni))
                    edges[7].append((ni, ai))

        for o, words in self.obj_neighbors.items():
            if o not in self.obj2id:
                continue
            oi = self.obj2id[o]
            for w in words:
                if w in self.neigh_obj2id:
                    ni = self.neigh_obj2id[w]
                    edges[7].append((oi, ni))
                    edges[7].append((ni, oi))

        for (a, o), phrases in self.comp_neighbors.items():
            if (a, o) not in self.comp2id:
                continue
            ci = self.comp2id[(a, o)]
            for ph in phrases:
                if ph in self.neigh_comp2id:
                    ni = self.neigh_comp2id[ph]
                    edges[7].append((ci, ni))
                    edges[7].append((ni, ci))

        # 转 tensor
        ei_list, et_list = [], []
        for rel_id, edge_list in edges.items():
            if edge_list:
                ei = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                et = torch.full((ei.shape[1],), rel_id, dtype=torch.long)
                ei_list.append(ei)
                et_list.append(et)
                print(f"[Graph] R{rel_id}({self.REL_NAMES[rel_id]}): {ei.shape[1]} 条边")

        self.edge_index = torch.cat(ei_list, dim=1)
        self.edge_type  = torch.cat(et_list)
        print(f"[Graph] 总边数: {self.edge_index.shape[1]}")

    def get_all_node_names(self):
        """返回所有节点文本名（按节点 ID 顺序），用于 CLIP 初始化"""
        names  = list(self.seen_attrs)
        names += list(self.seen_objs)
        names += [f"{a} {o}" for a, o in self.seen_comps]
        names += list(self.neigh_attrs)
        names += list(self.neigh_objs)
        names += list(self.neigh_comps)
        return names


# ─────────────────────────────────────────────────────────────
class RGCNModule(nn.Module):
    """支持动态层数的 RGCN，带基函数分解，防止参数爆炸"""

    def __init__(self, in_dim, hidden_dim, out_dim,
                 num_relations=8, num_bases=4, dropout=0.3, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.drop = nn.Dropout(dropout)
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            # 如果只有 1 层，直接映射 in_dim -> out_dim
            self.convs.append(RGCNConv(in_dim, out_dim, num_relations, num_bases=num_bases))
            self.norms.append(nn.LayerNorm(out_dim))
        else:
            # 第一层: in_dim -> hidden_dim
            self.convs.append(RGCNConv(in_dim, hidden_dim, num_relations, num_bases=num_bases))
            self.norms.append(nn.LayerNorm(hidden_dim))
            
            # 中间层: hidden_dim -> hidden_dim
            for _ in range(num_layers - 2):
                self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=num_bases))
                self.norms.append(nn.LayerNorm(hidden_dim))
                
            # 最后一层: hidden_dim -> out_dim
            self.convs.append(RGCNConv(hidden_dim, out_dim, num_relations, num_bases=num_bases))
            self.norms.append(nn.LayerNorm(out_dim))

    def forward(self, x, edge_index, edge_type):
        """
        x:          (N, in_dim)
        edge_index: (2, E)
        edge_type:  (E,)  每条边的关系 ID
        Returns:    (N, out_dim)
        """
        h = x
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index, edge_type)
            h = self.norms[i](h)
            
            # 最后一层不加 ReLU 和 Dropout，保留完整的特征空间分布
            if i < self.num_layers - 1:
                h = F.relu(h)
                h = self.drop(h)
                
        return h