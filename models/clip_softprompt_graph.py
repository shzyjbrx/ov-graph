import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import torch.distributed as dist

from .clip_softprompt import CLIPSoftPrompt
from .graph_module import HeteroGraphBuilder, RGCNModule

class MetaNet(nn.Module):
    def __init__(self, in_dim, emb_dim, n_meta_ctx=4, hidden_dim=256):
        super().__init__()
        self.n_meta_ctx = n_meta_ctx
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, emb_dim * n_meta_ctx),
        )

    def forward(self, h):
        N   = h.shape[0]
        out = self.net(h)                        
        return out.view(N, self.n_meta_ctx, -1)  

class CLIPSoftPromptGraph(CLIPSoftPrompt):
    def __init__(self, dset, cfg):
        super().__init__(dset, cfg)

        clip_dim   = self.emb_dim                              
        graph_dim  = getattr(cfg.MODEL, 'graph_dim',     512)
        n_meta     = getattr(cfg.MODEL, 'n_meta_ctx',      4)
        n_bases    = getattr(cfg.MODEL, 'graph_bases',     4)
        graph_drop = getattr(cfg.MODEL, 'graph_dropout', 0.3)
        max_k      = getattr(cfg.MODEL, 'max_neighbors',   5)
        graph_layers = getattr(cfg.MODEL, 'graph_layers',  2)

        use_llm_nel    = getattr(cfg.MODEL, 'use_llm_nel', False)
        neighbor_paths = None
        if use_llm_nel:
            llm_dir = getattr(cfg.MODEL, 'llm_nel_dir', '')
            neighbor_paths = {
                'attr': os.path.join(llm_dir, 'attr_neighbors.json'),
                'obj':  os.path.join(llm_dir, 'obj_neighbors.json'),
                'comp': os.path.join(llm_dir, 'comp_neighbors.json'),
            }
            print(f"[Graph] LLM 邻域扩展开启，路径: {llm_dir}")
        else:
            print("[Graph] 仅 Seen 节点图（use_llm_nel=False）")

        self.graph_builder = HeteroGraphBuilder(dset, neighbor_paths, max_k)
        gb = self.graph_builder

        self.rgcn = RGCNModule(
            in_dim=clip_dim, hidden_dim=graph_dim, out_dim=clip_dim,
            num_relations=HeteroGraphBuilder.NUM_RELATIONS,
            num_bases=n_bases, dropout=graph_drop,
            num_layers=graph_layers
        )

        self.n_meta_ctx  = n_meta
        self.meta_attr   = MetaNet(clip_dim, clip_dim, n_meta)
        self.meta_obj    = MetaNet(clip_dim, clip_dim, n_meta)
        self.meta_comp   = MetaNet(clip_dim, clip_dim, n_meta)

        print("[Graph] 初始化图节点特征 H(0)...")
        self._init_node_features()

        seen_a2local = {a: i for i, a in enumerate(gb.seen_attrs)}
        seen_o2local = {o: i for i, o in enumerate(gb.seen_objs)}
        pair_la = [seen_a2local[a] for a, o in gb.seen_comps]
        pair_lo = [seen_o2local[o] for a, o in gb.seen_comps]
        self.register_buffer('pair_local_a', torch.tensor(pair_la, dtype=torch.long))
        self.register_buffer('pair_local_o', torch.tensor(pair_lo, dtype=torch.long))

        self.register_buffer('graph_edge_index', gb.edge_index)
        self.register_buffer('graph_edge_type',  gb.edge_type)

        self._nA = len(gb.seen_attrs)
        self._nO = len(gb.seen_objs)
        self._nC = len(gb.seen_comps)

    @torch.no_grad()
    def _init_node_features(self):
        """🌟 绝对安全的 CPU 计算，彻底防止多进程 DataLoader 崩溃"""
        from tqdm import tqdm

        dset_name = getattr(self.cfg.DATASET, 'name', 'dataset')
        use_llm_nel = getattr(self.cfg.MODEL, 'use_llm_nel', False)
        cache_path = f"cache_{dset_name}_graph_H0_llm_{use_llm_nel}.pt"
        rank = int(os.environ.get('RANK', 0))

        if rank == 0:
            if not os.path.exists(cache_path):
                print(f"[Graph] 未发现图特征缓存，Rank 0 开始在 CPU 上独立计算...")
                print(f"[Graph] ⚠️ 预计需要 30-60 秒，请耐心等待，这能彻底避免多进程崩溃！")

                node_names = self.graph_builder.get_all_node_names()
                all_feats  = []
                BATCH      = 128

                for i in tqdm(range(0, len(node_names), BATCH), desc="Init H(0) (CPU)"):
                    batch_names = node_names[i: i + BATCH]
                    texts  = [f"a photo of {n.replace('_', ' ')}" for n in batch_names]
                    
                    # 强制在 CPU 上运行，杜绝一切 CUDA 上下文污染
                    tokens = clip.tokenize(texts, truncate=True)
                    x = self.token_embedding(tokens).float()
                    x = x + self.positional_embedding.float()
                    x = x.permute(1, 0, 2)
                    x = self.transformer(x)
                    x = x.permute(1, 0, 2)
                    x = self.ln_final(x).float()

                    eos = tokens.argmax(dim=-1)
                    x = x[torch.arange(x.shape[0]), eos] @ self.text_projection.float()
                    
                    all_feats.append(F.normalize(x, dim=-1))

                H0 = torch.cat(all_feats, dim=0)
                torch.save(H0, cache_path)
                print(f"\n[Graph] Rank 0 图特征保存完成！")
            else:
                print(f"[Graph] 发现已存在的图特征缓存: {cache_path}")

        # 强制其他卡原地等待，直到主卡把特征存完硬盘
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        # 所有进程同步加载，保证 buffer 完美注册
        H0 = torch.load(cache_path, map_location='cpu')
        self.register_buffer('node_feats_init', H0)
        
        if rank == 0:
            print("[Graph] 所有进程图特征 Buffer 均已完美同步并注册！")

    def _propagate(self):
        H = self.rgcn(
            self.node_feats_init,
            self.graph_edge_index,
            self.graph_edge_type,
        )    
        return H

    def _split_H(self, H):
        nA, nO, nC = self._nA, self._nO, self._nC
        h_attr = H[:nA]
        h_obj  = H[nA: nA + nO]
        h_comp = H[nA + nO: nA + nO + nC]
        return h_attr, h_obj, h_comp

    def _encode_text_with_graph(self, ctx, class_embeds, base_anchor, meta_tokens, mini_batch=256):
        N = class_embeds.shape[0]
        results = []

        for i in range(0, N, mini_batch):
            ce   = class_embeds[i: i + mini_batch]   
            ba   = base_anchor[i: i + mini_batch]     
            mt   = meta_tokens[i: i + mini_batch]     
            b    = ce.shape[0]
            D    = self.emb_dim
            n_ctx  = self.n_ctx
            n_meta = self.n_meta_ctx

            with torch.no_grad():
                sos_emb = self.token_embedding(self.sos_token).float()
                eos_emb = self.token_embedding(self.eos_token).float()

            sos_exp = sos_emb.expand(b, -1, -1)               
            ctx_exp = ctx.unsqueeze(0).expand(b, -1, -1).float()  
            cls_exp = ce.unsqueeze(1)                          
            eos_exp = eos_emb.expand(b, -1, -1)               

            prefix  = torch.cat([sos_exp, ctx_exp, mt, cls_exp, eos_exp], dim=1)
            seq_len = prefix.shape[1]   

            if seq_len < 77:
                pad = torch.zeros(b, 77 - seq_len, D, device=prefix.device, dtype=prefix.dtype)
                x = torch.cat([prefix, pad], dim=1)
            else:
                x = prefix[:, :77, :]

            x = x + self.positional_embedding.float()
            x = x.permute(1, 0, 2)
            x = self.transformer(x)
            x = x.permute(1, 0, 2).float()
            x = self.ln_final(x)

            eos_pos = min(1 + n_ctx + n_meta + 1, 76)
            learned = x[:, eos_pos] @ self.text_projection.float()
            learned = F.normalize(learned, dim=-1)

            combined = ba + self.alpha * learned
            results.append(F.normalize(combined, dim=-1))

        return torch.cat(results, dim=0)

    @torch.no_grad()
    def _retrieve_graph_feat(self, concept_name, H, node_type, topk=3):
        text   = f"a photo of {concept_name.replace('_', ' ')}"
        tokens = clip.tokenize([text], truncate=True).to(H.device)

        x = self.token_embedding(tokens).float()
        x = x + self.positional_embedding.float()
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).float()
        eos = tokens.argmax(dim=-1)
        q = x[0, eos[0]] @ self.text_projection.float()
        q = F.normalize(q, dim=-1)   

        nA, nO = self._nA, self._nO
        if node_type == 'attr':
            cand_H = H[:nA]
        else:
            cand_H = H[nA: nA + nO]

        sims    = F.normalize(cand_H, dim=-1) @ q   
        topk    = min(topk, sims.shape[0])
        vals, idxs = sims.topk(topk)
        weights = F.softmax(vals * 10.0, dim=0)
        return (cand_H[idxs] * weights.unsqueeze(1)).sum(0)   

    def forward(self, batch):
        if self.training:
            self.test_text_features = None
            return self._train_forward_graph(batch)

        if self.test_text_features is None:
            with torch.no_grad():
                self.test_text_features = self._precompute_test_feats()

        return self._val_forward(batch)

    def _train_forward_graph(self, batch):
        imgs      = batch['img']
        p_idx_tr  = batch['pair']    
        a_idx     = batch['attr']    
        o_idx     = batch['obj']     

        v = self._encode_visual(imgs)   

        H = self._propagate()           
        h_attr, h_obj, h_comp = self._split_H(H)

        meta_a = self.meta_attr(h_attr)                     
        t_a = self._encode_text_with_graph(
            self.ctx_attr,
            self.attr_embeds[self.tr_attr_idx],             
            self.base_attr_all[self.tr_attr_idx],           
            meta_a,
        )   

        meta_o = self.meta_obj(h_obj)                       
        t_o = self._encode_text_with_graph(
            self.ctx_obj,
            self.obj_embeds[self.tr_obj_idx],
            self.base_obj_all[self.tr_obj_idx],
            meta_o,
        )   

        pair_ha = h_attr[self.pair_local_a]                 
        pair_ho = h_obj[self.pair_local_o]                  
        h_pair  = (pair_ha + pair_ho) / 2.0                 

        meta_c = self.meta_comp(h_pair)                     

        pair_ae = self.attr_embeds[self.train_pair_attr_indices]  
        pair_oe = self.obj_embeds[self.train_pair_obj_indices]
        t_c = self._encode_text_with_graph(
            self.ctx_comp,
            (pair_ae + pair_oe) / 2.0,
            self.base_pair_tr,
            meta_c,
        )   

        scale    = self.logit_scale.exp()
        logits_c = scale * (v @ t_c.T)   
        logits_a = scale * (v @ t_a.T)   
        logits_o = scale * (v @ t_o.T)   

        loss_c = F.cross_entropy(logits_c, p_idx_tr)
        loss_a = F.cross_entropy(logits_a, a_idx)
        loss_o = F.cross_entropy(logits_o, o_idx)

        loss = (loss_c
                + self.cfg.MODEL.w_loss_attr * loss_a
                + self.cfg.MODEL.w_loss_obj  * loss_o)

        w_neigh = getattr(self.cfg.MODEL, 'w_loss_neigh', 0.0)
        if w_neigh > 0.0:
            neigh_loss = self._compute_neigh_smooth_loss(H)
            loss = loss + w_neigh * neigh_loss

        return {
            'loss_total': loss,
            'acc_pair':   (logits_c.argmax(1) == p_idx_tr).float().mean(),
        }

    def _compute_neigh_smooth_loss(self, H):
        gb     = self.graph_builder
        target = torch.tensor(0.7, device=H.device)
        loss   = torch.tensor(0.0, device=H.device)
        count  = 0

        for a, words in gb.attr_neighbors.items():
            if a not in gb.attr2id:
                continue
            h_seen = H[gb.attr2id[a]]
            for w in words:
                if w in gb.neigh_attr2id:
                    h_neigh = H[gb.neigh_attr2id[w]]
                    sim = F.cosine_similarity(h_seen.unsqueeze(0),
                                             h_neigh.unsqueeze(0))
                    loss  = loss + (sim - target).pow(2)
                    count += 1

        for o, words in gb.obj_neighbors.items():
            if o not in gb.obj2id:
                continue
            h_seen = H[gb.obj2id[o]]
            for w in words:
                if w in gb.neigh_obj2id:
                    h_neigh = H[gb.neigh_obj2id[w]]
                    sim = F.cosine_similarity(h_seen.unsqueeze(0),
                                             h_neigh.unsqueeze(0))
                    loss  = loss + (sim - target).pow(2)
                    count += 1

        return loss / max(count, 1)

    @torch.no_grad()
    def _precompute_test_feats(self):
        H = self._propagate()
        gb = self.graph_builder
        test_pairs = self.all_pairs1

        cls_list, base_list, meta_list = [], [], []

        for (a, o) in test_pairs:
            ae = self.attr_embeds[self.dset.attr2idx[a]]
            oe = self.obj_embeds[self.dset.obj2idx[o]]
            cls_list.append((ae + oe) / 2.0)

            if a in gb.attr2id:
                ha = H[gb.attr2id[a]]
            else:
                ha = self._retrieve_graph_feat(a, H, 'attr')

            if o in gb.obj2id:
                ho = H[gb.obj2id[o]]
            else:
                ho = self._retrieve_graph_feat(o, H, 'obj')

            meta_list.append((ha + ho) / 2.0)

        cls_all  = torch.stack(cls_list).to(H.device)   
        h_all    = torch.stack(meta_list).to(H.device)  
        meta_all = self.meta_comp(h_all)                 

        pair_names = [f"{a} {o}" for a, o in test_pairs]
        base_all   = self._compute_manual_feats_live(pair_names).to(H.device)

        return self._encode_text_with_graph(
            self.ctx_comp, cls_all, base_all, meta_all
        )