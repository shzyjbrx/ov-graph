import os
import re
import sys
import json
import argparse
from typing import List, Dict

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    p = argparse.ArgumentParser(description="Generate English neighborhood vocab using LLM")
    p.add_argument("--data_root",      type=str, default="/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Base/data/mit-states")
    p.add_argument("--save_dir",       type=str, default="./llm_nel_gen/neighbors_data")
    p.add_argument("--model_id",       type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--max_retries",    type=int, default=4) # 增加重试次数
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature",    type=float, default=0.3) # 稍微提高温度，避免模型卡死在错误回答上
    return p.parse_args()

def load_vocab(data_root: str):
    split_dir = os.path.join(data_root, "compositional-split-natural")
    attrs_set, objs_set, pairs_set = set(), set(), set()
    files = ["train_pairs.txt", "val_pairs.txt", "test_pairs.txt"]
    for fname in files:
        path = os.path.join(split_dir, fname)
        if not os.path.exists(path): continue
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    attrs_set.add(parts[0])
                    objs_set.add(parts[1])
                    pairs_set.add((parts[0], parts[1]))
    return sorted(attrs_set), sorted(objs_set), sorted(pairs_set)

def clean_name(name: str) -> str:
    return name.replace("_", " ").replace(".", " ").strip()

# ==========================================
# 强化过滤：只允许英文字母、数字、空格和连字符
# ==========================================
def is_valid_english(text: str) -> bool:
    # 过滤掉包含 \u 或者 中文字符 的脏数据
    if not text or len(text) < 2: return False
    return bool(re.match(r'^[A-Za-z0-9\s\-\']+$', text))

def build_en_prompt(node_type: str, concept: str) -> list:
    name = clean_name(concept)
    if node_type == "attr":
        system = "You are a linguistic expert. Output MUST be ONLY English words in JSON format. NO Chinese."
        user = f"Target Attribute: {name}\n1. 3 synonyms.\n2. 2 fine-grained states.\nFormat:\n{{\"synonyms\": [\"a\", \"b\", \"c\"], \"fine_grained\": [\"d\", \"e\"]}}"
    elif node_type == "obj":
        system = "You are a visual reasoning engine. Output MUST be ONLY English words in JSON format. NO Chinese."
        user = f"Target Object: {name}\n1. 2 hypernyms.\n2. 4 visual siblings.\nFormat:\n{{\"hypernyms\": [\"a\", \"b\"], \"visual_siblings\": [\"c\", \"d\", \"e\", \"f\"]}}"
    else:
        attr_part, obj_part = name.split(' ', 1) if ' ' in name else (name, "")
        system = "You are a multi-modal visual description expert. Output MUST be ONLY English words in JSON format. NO Chinese."
        user = f"Target Composition: {name}\n1. 2 Holistic Synonyms.\n2. 2 Attr Substituted (keep {obj_part}).\n3. 2 Obj Substituted (keep {attr_part}).\nFormat:\n{{\"holistic_synonyms\": [\"a\", \"b\"], \"attr_substituted\": [\"c\", \"d\"], \"obj_substituted\": [\"e\", \"f\"]}}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

# ==========================================
# 暴力解析：兼容标准 JSON 与残缺 JSON
# ==========================================
def extract_robust(response: str) -> list:
    flat_list = []
    
    # 尝试 1: 标准 JSON 解析
    start, end = response.find('{'), response.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(response[start:end+1])
            for val in data.values():
                if isinstance(val, list):
                    flat_list.extend([str(v).lower().strip() for v in val])
        except json.JSONDecodeError:
            pass

    # 尝试 2: 如果标准解析失败或结果为空，用正则暴力提取双引号内的内容
    if not flat_list:
        matches = re.findall(r'"([^"]+)"', response)
        # 排除掉 JSON 的 keys
        exclude_keys = {"synonyms", "fine_grained", "hypernyms", "visual_siblings", "holistic_synonyms", "attr_substituted", "obj_substituted"}
        flat_list = [m.lower().strip() for m in matches if m.lower().strip() not in exclude_keys]

    # 终极过滤与去重
    final_list = [w for w in flat_list if is_valid_english(w)]
    final_list = list(dict.fromkeys(final_list)) # 保持顺序去重
    
    return final_list

def generate_and_parse(tokenizer, model, node_type, concept, args):
    for attempt in range(args.max_retries):
        messages = build_en_prompt(node_type, concept)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "{"
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            ids = model.generate(
                **inputs, 
                max_new_tokens=args.max_new_tokens, 
                temperature=args.temperature + (attempt * 0.1), # 每次重试稍微提高温度以摆脱死循环
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        response = "{" + tokenizer.batch_decode(ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        
        extracted_words = extract_robust(response)
        if len(extracted_words) >= 3: # 只要成功抓到3个以上的合法英文词，就认为成功
            return extracted_words
            
    print(f"\n[Warning] Failed to generate for: {concept}")
    return ["[FAILED]"]

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    attrs, objs, pairs = load_vocab(args.data_root)
    
    print(f"Loading Model: {args.model_id}")
    # 强制加上 trust_remote_code
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

    tasks = [
        (attrs, "attr", "attr_neighbors.json"), 
        (objs, "obj", "obj_neighbors.json"), 
        (pairs, "comp", "comp_neighbors.json")
    ]
    
    for node_list, node_type, filename in tasks:
        out_path = os.path.join(args.save_dir, filename)
        results = {}
        
        # [断点续跑逻辑]：读取已有的 JSON，不清空之前成功的成果
        if os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as f:
                try:
                    results = json.load(f)
                    print(f"\nLoaded {len(results)} existing entries from {filename}")
                except json.JSONDecodeError:
                    print(f"\nWarning: {filename} is corrupted. Starting fresh.")
                    results = {}

        print(f"\nProcessing {node_type} nodes...")
        for item in tqdm(node_list):
            key = f"{item[0]} {item[1]}" if isinstance(item, tuple) else item
            
            # 判断是否需要 (重新) 生成
            needs_gen = False
            if key not in results:
                needs_gen = True
            else:
                val = results[key]
                # 如果之前是 FAILED，或者是空列表，或者包含乱码
                if not isinstance(val, list) or len(val) == 0 or val[0] == "[FAILED]":
                    needs_gen = True
                else:
                    for w in val:
                        if not is_valid_english(w):
                            needs_gen = True
                            break

            if needs_gen:
                results[key] = generate_and_parse(tokenizer, model, node_type, key, args)
                # 实时保存，防止中断
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                    
        print(f"Finished {node_type}. Saved to: {out_path}")

if __name__ == "__main__":
    main()