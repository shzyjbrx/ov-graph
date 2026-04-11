import os
import random
import torch
import clip
import argparse
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader

from config import cfg
from dataset import CompositionDataset
from evaluator import Evaluator
from models.clip_softprompt_graph import CLIPSoftPromptGraph

def classify_pair(attr_idx, obj_idx, evaluator):
    """
    根据 Evaluator 中的划分，将 (attr, obj) 分类为 5 种泛化情形
    """
    pair = (attr_idx, obj_idx)
    if pair in evaluator.sa_so:
        return 'AO'
    elif pair in evaluator.sa_so_u:
        return '(AO)*'
    elif pair in evaluator.ua_so:
        return 'A*O'
    elif pair in evaluator.sa_uo:
        return 'AO*'
    elif pair in evaluator.ua_uo:
        return 'A*O*'
    else:
        return 'Unknown'

def main():
    parser = argparse.ArgumentParser()
    # 填入你平时训练使用的基础 yml 文件
    parser.add_argument('--cfg', type=str, default='config/mit_softprompt_graph.yml', help='path to config file')
    # 新增参数：每个类别抽样的数量
    parser.add_argument('--num_per_cat', type=int, default=50, help='Number of images to sample per category')
    # 新增参数：随机种子，保证每次抽样结果可复现
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    args = parser.parse_args()

    # 设置随机种子
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # =================================================================
    # 1. 加载基础配置并覆盖为日志中的最优参数
    # =================================================================
    cfg.merge_from_file(args.cfg)
    cfg.defrost() # 解冻配置以允许修改
    
    cfg.DATASET.name = 'mitstates'
    cfg.DATASET.dset_name = 'mit'
    cfg.DATASET.dset_split = 1  # 强制对齐数据集划分
    cfg.DATASET.root_dir = '/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Base/data/mit-states'
    cfg.DATASET.split_files_loc = 'Data_files/MIT'
    cfg.DATASET.splitname = 'compositional-split-natural'
    
    cfg.MODEL.name = 'clip_softprompt_graph'
    cfg.MODEL.n_ctx = 16
    cfg.MODEL.n_meta_ctx = 4
    cfg.MODEL.graph_bases = 4
    cfg.MODEL.graph_dim = 512
    cfg.MODEL.graph_dropout = 0.3
    cfg.MODEL.graph_layers = 2
    cfg.MODEL.max_neighbors = 5 # 强制对齐每个节点的邻居上限
    cfg.MODEL.use_llm_nel = True
    cfg.MODEL.llm_nel_dir = '/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/ov-czsl/llm_nel_gen/mit-states_neighbors'
    cfg.MODEL.use_nel_data = False
    
    cfg.MODEL.use_attr_loss = False
    cfg.MODEL.use_obj_loss = False
    cfg.MODEL.use_composed_pair_loss = False
    cfg.MODEL.use_emb_pair_loss = False
    
    cfg.TRAIN.clip_type = 'ViT-L/14'
    cfg.TRAIN.test_batch_size = 512
    cfg.TRAIN.use_precomputed_features = False # 解决 AttributeError
    
    cfg.freeze() # 重新冻结配置

    # =================================================================
    # 2. 准备数据集、Evaluator 与 抽样逻辑
    # =================================================================
    print("==> Loading datasets and evaluator...")
    trainset = CompositionDataset(phase='train', split=cfg.DATASET.splitname, cfg=cfg)
    testset = CompositionDataset(phase='test', split=cfg.DATASET.splitname, cfg=cfg)
    evaluator = Evaluator(testset, cfg)
    closed_mask = evaluator.closed_mask.to(device)

    print("==> Classifying testset and sampling images...")
    # 按照类别收集所有测试集图片的索引
    cat_to_indices = defaultdict(list)
    for i in tqdm(range(len(testset.data)), desc="Categorizing"):
        img_path, attr_str, obj_str = testset.data[i]
        attr_idx = testset.attr2idx[attr_str]
        obj_idx = testset.obj2idx[obj_str]
        
        category = classify_pair(attr_idx, obj_idx, evaluator)
        cat_to_indices[category].append(i)

    # 从每个类别中随机抽取指定数量的索引 (最多取 num_per_cat 个)
    sampled_indices = []
    target_categories = ['AO', '(AO)*', 'A*O', 'AO*', 'A*O*']
    
    for cat in target_categories:
        available_indices = cat_to_indices.get(cat, [])
        sample_size = min(args.num_per_cat, len(available_indices))
        if sample_size > 0:
            sampled_indices.extend(random.sample(available_indices, sample_size))
            print(f"    - Sampled {sample_size} images for category {cat}")
        else:
            print(f"    - Warning: 0 images found for category {cat}")

    # 使用 Subset 创建仅包含抽样数据的 DataLoader (batch_size 依然为 1 方便打印)
    sampled_testset = Subset(testset, sampled_indices)
    testloader = DataLoader(sampled_testset, batch_size=1, shuffle=False, num_workers=4)

    # =================================================================
    # 3. 加载你的最优模型
    # =================================================================
    print("\n==> Loading optimal model (Graph Softprompt)...")
    optimal_model = CLIPSoftPromptGraph(trainset, cfg).to(device)
    model_path = '/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/ov-graph/checkpoints/mit/Graph/1183025/mit_softprompt_graph_124/best_model.pth'
    
    state_dict = torch.load(model_path, map_location=device)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    optimal_model.load_state_dict(state_dict)
    optimal_model.eval()

    val_attrs, val_objs = zip(*testset.pairs)
    val_attrs = [testset.attr2idx[attr] for attr in val_attrs]
    val_objs = [testset.obj2idx[obj] for obj in val_objs]
    optimal_model.val_attrs = torch.LongTensor(val_attrs).to(device)
    optimal_model.val_objs = torch.LongTensor(val_objs).to(device)
    optimal_model.val_pairs = testset.pairs

    # =================================================================
    # 4. 加载原始 CLIP 模型用于对比
    # =================================================================
    print("==> Loading original CLIP model...")
    clip_path = '/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Base/checkpoints/ViT-L-14.pt'
    clip_model, preprocess = clip.load(clip_path, device=device)
    clip_model.eval()

    pairs_str = testset.pairs 
    prompts = [f"a photo of a {pair[0]} {pair[1]}" for pair in pairs_str]
    text_tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        clip_text_features = clip_model.encode_text(text_tokens)
        clip_text_features = clip_text_features / clip_text_features.norm(dim=-1, keepdim=True)

    # =================================================================
    # 5. 开始推理并收集分类结果
    # =================================================================
    results_by_category = defaultdict(list)

    print("\n==> Starting Inference on Sampled Test Set...")
    with torch.no_grad():
        for data in tqdm(testloader, desc='Inferencing'):
            for k in data:
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].to(device)

            images = data['img']
            attr_idx = data['attr'][0].item()
            obj_idx = data['obj'][0].item()
            img_name = data['img_name'][0]  # 这就是图片的文件路径/文件名
            
            gt_str = f"{testset.attrs[attr_idx]} {testset.objs[obj_idx]}"
            category = classify_pair(attr_idx, obj_idx, evaluator)
            
            # 纯 CLIP 预测
            image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            clip_logits = clip_model.logit_scale.exp() * image_features @ clip_text_features.T
            
            clip_logits = clip_logits.float() # 解决 float16 overflow 报错
            clip_logits[0, ~closed_mask] = -1e10 # Closed-world evaluation
            clip_pred_idx = clip_logits.argmax(dim=-1).item()
            clip_pred_str = f"{pairs_str[clip_pred_idx][0]} {pairs_str[clip_pred_idx][1]}"

            # ==================================
            # 最优模型预测
            # ==================================
            out = optimal_model(data)
            raw_scores_dict = out['scores']
            
            # 🚀 修复点：将字典格式的得分根据测试集 pairs 的顺序拼接成 [1, num_pairs] 的 Tensor
            model_scores = torch.stack([raw_scores_dict[pair] for pair in pairs_str], dim=1)
            
            model_scores = model_scores.float() # 防御性转换，防溢出
            model_scores[0, ~closed_mask] = -1e10 # Closed-world evaluation
            model_pred_idx = model_scores.argmax(dim=-1).item()
            model_pred_str = f"{pairs_str[model_pred_idx][0]} {pairs_str[model_pred_idx][1]}"

            # 存储格式：强调 img_name (路径)
            log_line = f"Path: {img_name:<30} | GT: {gt_str:<20} | CLIP: {clip_pred_str:<20} | Model: {model_pred_str:<20}"
            results_by_category[category].append(log_line)

    # =================================================================
    # 6. 分门别类打印输出到日志
    # =================================================================
    print("\n" + "#" * 120)
    print(" " * 45 + "SAMPLED PREDICTION RESULTS BY CATEGORY")
    print("#" * 120)

    for cat in target_categories:
        samples = results_by_category.get(cat, [])
        print(f"\n==================== CATEGORY: {cat} ({len(samples)} samples) ====================")
        for line in samples:
            print(line)
        
    print("\n==> Inference completed successfully.")

if __name__ == '__main__':
    main()