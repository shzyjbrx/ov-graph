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
        """🚀 移至 GPU 的特征预计算，大幅提速"""
        from tqdm import tqdm

        dset_name = getattr(self.cfg.DATASET, 'name', 'dataset')
        use_llm_nel = getattr(self.cfg.MODEL, 'use_llm_nel', False)
        cache_path = f"cache_{dset_name}_graph_H0_llm_{use_llm_nel}.pt"
        rank = int(os.environ.get('RANK', 0))

        if rank == 0:
            if not os.path.exists(cache_path):
                print(f"[Graph] 未发现图特征缓存，Rank 0 开始在 GPU 上高速计算...")
                
                # 1. 获取当前 Rank 对应的 GPU 设备
                device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
                
                # 2. 临时将需要用到的 CLIP 文本组件放到 GPU 上
                self.token_embedding.to(device)
                self.transformer.to(device)
                self.ln_final.to(device)
                pos_emb = self.positional_embedding.to(device).float()
                text_proj = self.text_projection.to(device).float()

                node_names = self.graph_builder.get_all_node_names()
                all_feats  = []
                BATCH      = 512  # GPU 并行能力强，Batch 调大加速

                for i in tqdm(range(0, len(node_names), BATCH), desc=f"Init H(0) (GPU)"):
                    batch_names = node_names[i: i + BATCH]
                    texts  = [f"a photo of {n.replace('_', ' ')}" for n in batch_names]
                    
                    # 3. 将输入的 token 移至 GPU
                    tokens = clip.tokenize(texts, truncate=True).to(device)
                    x = self.token_embedding(tokens).float()
                    x = x + pos_emb
                    x = x.permute(1, 0, 2)
                    x = self.transformer(x)
                    x = x.permute(1, 0, 2)
                    x = self.ln_final(x).float()

                    eos = tokens.argmax(dim=-1)
                    x = x[torch.arange(x.shape[0]), eos] @ text_proj
                    
                    # 4. 计算完毕后一定要 .cpu() 移回内存，防止缓存大图谱时爆显存
                    all_feats.append(F.normalize(x, dim=-1).cpu())

                H0 = torch.cat(all_feats, dim=0)
                torch.save(H0, cache_path)
                print(f"\n[Graph] Rank 0 图特征 (GPU加速) 保存完成！")
            else:
                print(f"[Graph] 发现已存在的图特征缓存: {cache_path}")

        # 强制其他卡原地等待，直到主卡把特征存完硬盘
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        # 所有进程同步加载，保证 buffer 完美注册 (强制加载到 cpu，稍后外部 model.to(device) 会自动搬运)
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
        import torch.utils.checkpoint as cp  # 引入检查点机制
        
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
            
            # ==========================================================
            # 核心修改：使用 Checkpoint 抹除 Transformer 的显存累积
            # ==========================================================
            # 只有在需要求导时（即训练时）才使用 checkpoint
            if x.requires_grad:
                x = cp.checkpoint(self.transformer, x, use_reentrant=False)
            else:
                x = self.transformer(x)
            # ==========================================================
            
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
            # 修改：传入生成的文本特征 (t_a, t_o, t_c) 和当前批次索引 (a_idx, o_idx, p_idx_tr)
            neigh_loss = self._compute_paper_neigh_smooth_loss(
                H, t_a, t_o, t_c, a_idx, o_idx, p_idx_tr
            )
            loss = loss + w_neigh * neigh_loss

        return {
            'loss_total': loss,
            'acc_pair':   (logits_c.argmax(1) == p_idx_tr).float().mean(),
        }

    def _compute_paper_neigh_smooth_loss(self, H, t_a_all, t_o_all, t_c_all, a_idx_batch, o_idx_batch, p_idx_batch):
        """
        严格遵循论文公式 (4-15), (4-16), (4-17) 的邻域对齐损失实现。
        """
        gb = self.graph_builder
        # H0 是最初使用冻结 CLIP 提取的零样本语义特征，用于计算平滑权重 w
        H0 = self.node_feats_init 
        
        loss_a = torch.tensor(0.0, device=H.device)
        loss_o = torch.tensor(0.0, device=H.device)
        loss_c = torch.tensor(0.0, device=H.device)
        
        batch_size = len(a_idx_batch)
        
        for i in range(batch_size):
            # ================= 1. 属性分支 (对应公式 4-16) =================
            a_id_local = a_idx_batch[i].item()
            a_name = gb.seen_attrs[a_id_local]
            a_node_id = gb.attr2id[a_name]
            t_a = t_a_all[a_id_local]  # 已见概念的软提示特征
            
            if a_name in gb.attr_neighbors:
                for w in gb.attr_neighbors[a_name]:
                    if w in gb.neigh_attr2id:
                        n_node_id = gb.neigh_attr2id[w]
                        # 计算基础文本相似度作为平滑权重 w (点积即余弦，因为 H0 已作 L2 归一化)
                        w_sim = torch.dot(H0[a_node_id], H0[n_node_id]).detach()
                        w_sim = torch.clamp(w_sim, min=0.0)  # 滤除极少数的负相关噪声
                        
                        # 异构图中结构化的邻域节点特征 (进行 L2 归一化以匹配 t_a 的尺度)
                        t_n = F.normalize(H[n_node_id], dim=0) 
                        # MSE 距离：|| t_a - t_n ||_2^2
                        loss_a = loss_a + w_sim * torch.sum((t_a - t_n) ** 2)
                        
            # ================= 2. 物体分支 (对应公式 4-17) =================
            o_id_local = o_idx_batch[i].item()
            o_name = gb.seen_objs[o_id_local]
            o_node_id = gb.obj2id[o_name]
            t_o = t_o_all[o_id_local]
            
            if o_name in gb.obj_neighbors:
                for w in gb.obj_neighbors[o_name]:
                    if w in gb.neigh_obj2id:
                        n_node_id = gb.neigh_obj2id[w]
                        w_sim = torch.dot(H0[o_node_id], H0[n_node_id]).detach()
                        w_sim = torch.clamp(w_sim, min=0.0)
                        
                        t_n = F.normalize(H[n_node_id], dim=0)
                        loss_o = loss_o + w_sim * torch.sum((t_o - t_n) ** 2)
                        
            # ================= 3. 组合分支 (对应公式 4-15) =================
            p_id_local = p_idx_batch[i].item()
            c_name_tuple = gb.seen_comps[p_id_local]
            c_node_id = gb.comp2id[c_name_tuple]
            t_c = t_c_all[p_id_local]
            
            if c_name_tuple in gb.comp_neighbors:
                for ph in gb.comp_neighbors[c_name_tuple]:
                    if ph in gb.neigh_comp2id:
                        n_node_id = gb.neigh_comp2id[ph]
                        w_sim = torch.dot(H0[c_node_id], H0[n_node_id]).detach()
                        w_sim = torch.clamp(w_sim, min=0.0)
                        
                        t_n = F.normalize(H[n_node_id], dim=0)
                        loss_c = loss_c + w_sim * torch.sum((t_c - t_n) ** 2)
                        
        # 批次内求平均并相加 (对应公式 4-18：L_nel = L_nel^a + L_nel^o + L_nel^c)
        return (loss_a + loss_o + loss_c) / batch_size

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