#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
generate_synthetic_fakes.py  # 利用本地 Ollama 和已有 real/fake 样本合成额外 fake reviews 并生成新的训练集

Usage:
    python generate_synthetic_fakes.py \
        --in_file data/reviews_with_labels.jsonl \
        --out_fake_file data/synthetic_fake_reviews.jsonl \
        --out_train_file data/train_augmented.jsonl \
        --ratio 1.0 \
        --model llama2 \
        --product_name "Sephora skincare product"
"""

import argparse  # argument parsing  # 解析命令行参数
import json  # json read/write  # 读写 JSON
import random  # random choice & shuffle  # 随机选择与打乱
from pathlib import Path  # file path helper  # 处理文件路径
import copy  # deep copy records  # 深拷贝字典对象

from tqdm import tqdm  # progress bar  # 进度条
import ollama  # local LLM client  # 本地 Ollama 客户端


# ------------------------------
# Helpers for JSONL I/O
# ------------------------------

def load_jsonl(path):  # load jsonl file  # 加载 jsonl 文件
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def save_jsonl(path, records):  # save list of dict to jsonl  # 将字典列表保存为 jsonl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ------------------------------
# Helpers for extracting fields
# ------------------------------

def get_field(rec, key, default=None):  # get field from rec or rec["meta"]  # 从主字典或 meta 中取字段
    if key in rec and rec[key] not in (None, ""):
        return rec[key]
    meta = rec.get("meta", {})
    if isinstance(meta, dict) and key in meta and meta[key] not in (None, ""):
        return meta[key]
    return default


def extract_review_text(rec):  # extract review_text string  # 提取评论文本
    txt = get_field(rec, "review_text", None)
    if txt:
        return str(txt)
    # fallback to context  # 回退到 context 字段
    ctx = rec.get("context", "")
    return str(ctx)


def extract_product_name(rec, fallback="a skincare product"):  # get product name  # 获取商品名称
    name = get_field(rec, "product_name", None)
    if name:
        return str(name)
    return fallback


def safe_float(x, default=None):  # safe float cast  # 安全转换为 float
    try:
        return float(x)
    except Exception:
        return default


# ------------------------------
# Split real / fake from labeled data
# ------------------------------

def split_real_fake(records, label_field="pseudo_label_v2"):  # split records by label  # 按标签拆分样本
    real, fake = [], []
    for r in records:
        label = r.get(label_field) or r.get("pseudo_label") or r.get("label")
        if label in ("real", 0, "0"):
            real.append(r)
        elif label in ("fake", 1, "1"):
            fake.append(r)
    return real, fake


# ------------------------------
# Prompt + Ollama generation
# ------------------------------

def build_prompt(real_review_text, fake_style_text, product_name):  # build english prompt  # 构造英文提示词
    real_review_text = real_review_text.replace("\n", " ").strip()
    fake_style_text = fake_style_text.replace("\n", " ").strip()

    prompt = f"""
You are helping generate synthetic low-quality or fake-looking product reviews for a machine learning dataset.

Here is a REAL user review for the product "{product_name}":
---
{real_review_text}
---

Here is an EXAMPLE of a low-quality / fake-looking review style:
---
{fake_style_text}
---

Now write ONE NEW fake-looking review for the SAME product "{product_name}".

Requirements:
- Use exaggerated or promotional language.
- Use an extreme sentiment (either very positive or very negative).
- Keep it short to medium length (about 40–120 words).
- Do NOT mention that you are an AI or that this is synthetic or fake.
- Do NOT talk about writing reviews, datasets, or training.
- The text should sound like a normal user review on an e-commerce site.

Return ONLY the review text, nothing else.
"""
    return prompt.strip()


def generate_fake_review_with_ollama(real_rec, fake_rec, model_name):  # call ollama to get one fake review  # 调用 Ollama 生成一条虚假评论
    real_text = extract_review_text(real_rec)
    fake_style_text = extract_review_text(fake_rec)
    product_name = extract_product_name(real_rec, fallback="a Sephora skincare product")

    prompt = build_prompt(real_text, fake_style_text, product_name)

    resp = ollama.chat(  # call local model  # 调用本地模型
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    # Ollama Python API returns {"message": {"role": "...", "content": "..."}}
    content = resp.get("message", {}).get("content", "").strip()
    return content or fake_style_text  # fallback to style text if empty  # 如为空则退回示例文本


# ------------------------------
# Generate synthetic fake records
# ------------------------------

def generate_synthetic_fake_records(real_reviews, fake_reviews, n_samples, model_name):  # generate many fake records  # 批量生成合成 fake 样本
    if not real_reviews:
        raise ValueError("No real reviews found. Cannot generate synthetic data.  # 没有 real 样本，无法生成合成数据")
    if not fake_reviews:
        # if no seed fake, reuse real as style, still works  # 如无 fake 样本，则用 real 作为风格参考
        fake_reviews = real_reviews

    synthetic_records = []

    for _ in tqdm(range(n_samples), desc="Generating synthetic fake reviews"):
        real_rec = random.choice(real_reviews)
        fake_rec = random.choice(fake_reviews)

        gen_text = generate_fake_review_with_ollama(real_rec, fake_rec, model_name)

        # create a new record based on real_rec skeleton  # 基于 real 样本结构创建新记录
        new_rec = copy.deepcopy(real_rec)
        # overwrite review-specific fields  # 重写评论相关字段
        new_rec["review_text"] = gen_text
        new_rec["pseudo_label_v2"] = "fake"
        new_rec["synthetic_source"] = "ollama"  # mark as synthetic  # 标记为合成样本

        # rating: extreme 1 or 5  # 评分：极端 1 或 5
        new_rating = random.choice([1, 5])
        new_rec["rating"] = new_rating
        if "meta" in new_rec and isinstance(new_rec["meta"], dict):
            new_rec["meta"]["rating"] = new_rating

        # is_recommended: random but consistent-ish  # 是否推荐：随机但略合理
        is_rec = 1 if new_rating >= 4 else 0
        new_rec["is_recommended"] = is_rec
        if "meta" in new_rec and isinstance(new_rec["meta"], dict):
            new_rec["meta"]["is_recommended"] = is_rec

        # helpfulness + feedback counts: make them small/low-quality  # 帮助度和反馈数：模拟低质量评论
        total_feedback = random.randint(0, 5)
        if total_feedback == 0:
            pos = 0
            neg = 0
            helpful = 0.0
        else:
            pos = random.randint(0, total_feedback)
            neg = total_feedback - pos
            helpful = pos / total_feedback if total_feedback > 0 else 0.0

        new_rec["total_feedback_count"] = total_feedback
        new_rec["total_pos_feedback_count"] = pos
        new_rec["total_neg_feedback_count"] = neg
        new_rec["helpfulness"] = helpful

        if "meta" in new_rec and isinstance(new_rec["meta"], dict):
            new_rec["meta"]["total_feedback_count"] = total_feedback
            new_rec["meta"]["total_pos_feedback_count"] = pos
            new_rec["meta"]["total_neg_feedback_count"] = neg
            new_rec["meta"]["helpfulness"] = helpful

        # review_title: simple headline  # 标题：简单标题
        new_rec["review_title"] = "Amazing product!" if new_rating == 5 else "Terrible experience"
        if "meta" in new_rec and isinstance(new_rec["meta"], dict):
            new_rec["meta"]["review_title"] = new_rec["review_title"]

        synthetic_records.append(new_rec)

    return synthetic_records


# ------------------------------
# Main: load -> generate -> save
# ------------------------------

def main():  # main entry  # 主入口函数
    parser = argparse.ArgumentParser(description="Generate synthetic fake reviews with Ollama and save augmented train jsonl.  # 使用 Ollama 生成合成 fake 评论并输出增强训练集")
    parser.add_argument("--in_file", type=str, required=True, help="Input labeled jsonl file (with real/fake labels).  # 已带 real/fake 标签的输入 jsonl 文件")
    parser.add_argument("--out_fake_file", type=str, required=True, help="Output jsonl file for synthetic fake reviews.  # 合成 fake 评论输出 jsonl 文件")
    parser.add_argument("--out_train_file", type=str, required=True, help="Output jsonl file for augmented training data.  # 增强后的训练集输出 jsonl 文件", default="data/train_augmented.jsonl")
    parser.add_argument("--ratio", type=float, default=1.0, help="Number of synthetic fakes = ratio * real_count.  # 合成 fake 数量 = ratio * real_count")
    parser.add_argument("--model", type=str, default="llama2", help="Ollama model name to use.  # 使用的 Ollama 模型名称")
    args = parser.parse_args()

    in_path = Path(args.in_file)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}  # 找不到输入文件")

    print(f"[INFO] loading labeled data from {in_path}")
    records = load_jsonl(in_path)

    real_reviews, fake_reviews = split_real_fake(records)
    real_count = len(real_reviews)
    fake_seed_count = len(fake_reviews)
    print(f"[INFO] real samples: {real_count}, seed fake samples: {fake_seed_count}")

    if real_count == 0:
        raise ValueError("No real samples found in input file.  # 输入文件中没有 real 样本")

    n_synth = int(args.ratio * real_count)
    if n_synth <= 0:
        print("[INFO] ratio <= 0, nothing to generate. Just copying input to out_train_file.  # ratio <= 0 不生成合成数据，直接复制输入")
        save_jsonl(args.out_train_file, records)
        return

    print(f"[INFO] generating {n_synth} synthetic fake reviews with model '{args.model}'")
    synthetic_fakes = generate_synthetic_fake_records(
        real_reviews=real_reviews,
        fake_reviews=fake_reviews,
        n_samples=n_synth,
        model_name=args.model,
    )

    print(f"[INFO] saving synthetic fake reviews to {args.out_fake_file}")
    save_jsonl(args.out_fake_file, synthetic_fakes)

    # merge original + synthetic, then shuffle  # 合并原始数据与合成数据并打乱
    all_train = records + synthetic_fakes
    random.shuffle(all_train)

    print(f"[INFO] saving augmented training data to {args.out_train_file}")
    save_jsonl(args.out_train_file, all_train)

    print("[INFO] done.")


if __name__ == "__main__":
    main()
