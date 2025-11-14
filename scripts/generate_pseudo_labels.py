#!/usr/bin/env python3
"""
generate_pseudo_labels.py
对 reviews_context.jsonl 中的每条记录做检索并基于简单规则打标签（真实 vs 虚假）
Usage:
    python scripts/generate_pseudo_labels.py --in_file data/reviews_context.jsonl --out_file data/reviews_with_labels.jsonl --top_k 3
"""

import argparse
import json
from build_retriever import Retriever
from tqdm import tqdm


def detect_factual_mismatch(review_context, retrieved_docs):
    """Detect simple factual mismatches between review text and retrieved product docs.  # 检测评论文本与检索到的商品文档之间的简单事实性矛盾"""
    text = review_context.lower()
    for doc in retrieved_docs:
        doc_text = doc["document"].lower()
        # check for fragrance contradiction: review mentions smell/fragrance but product is fragrance-free
        if ("smell" in text or "fragrance" in text) and "fragrance-free" in doc_text:
            return "fake"  # If contradiction is found, mark as fake

        # review mentions perfume/fragrance/smell and product text confirms fragrance (no fragrance-free)
        if (
                ("fragrance" in text or "perfume" in text or "smell" in text)
                and "fragrance" in doc_text
                and "fragrance-free" not in doc_text
        ):
            return "real"  # If consistent, mark as real

    return "real"  # If no contradiction found, assume real


def safe_float(x, default=None):
    """Safely convert to float.  # 安全地转换为 float"""
    try:
        return float(x)
    except (ValueError, TypeError):
        return default


def safe_int(x, default=None):
    """Safely convert to int.  # 安全地转换为 int"""
    try:
        return int(x)
    except (ValueError, TypeError):
        return default


def compute_fake_score(j):
    """Compute a heuristic fake score for this review.  # 计算该评论的假评论启发式得分"""
    ctx = j.get("context", "") or ""
    meta = j.get("meta", {}) or {}

    # Extracting fields safely  # 安全提取字段
    rating = safe_float(meta.get("rating"), None)
    is_recommended = safe_int(meta.get("is_recommended"), None)
    helpfulness = safe_float(meta.get("helpfulness"), None)
    total_feedback_count = safe_int(meta.get("total_feedback_count"), None)
    total_pos_feedback_count = safe_int(meta.get("total_pos_feedback_count"), None)
    total_neg_feedback_count = safe_int(meta.get("total_neg_feedback_count"), None)

    text = (ctx).strip()
    text_len = len(text)

    fake_score = 0

    # If necessary fields are missing, mark as unknown  # 如果缺少必要字段，标记为 unknown
    if rating is None or not text:
        return float('inf')  # Return high score to mark as fake (unknown)

    # Heuristic rules to detect fake reviews  # 启发式规则检测虚假评论
    if rating in [1.0, 5.0] and text_len < 20:
        fake_score += 2

    if helpfulness is not None and total_feedback_count is not None and total_feedback_count >= 3 and helpfulness < 0.3:
        fake_score += 1

    if total_feedback_count == 0 and rating in [1.0, 5.0] and text_len < 40:
        fake_score += 1

    if rating <= 2 and is_recommended == 1:
        fake_score += 1
    if rating >= 4 and is_recommended == 0:
        fake_score += 1

    return fake_score


def decide_final_label(fact_label, fake_score, j, fake_threshold=2):
    """Decide final pseudo label among real / fake  # 决策最终伪标签 real / fake"""
    ctx = j.get("context", "") or ""
    meta = j.get("meta", {}) or {}
    base_label = j.get("pseudo_label", None)

    # Extract rating from the meta information  # 从 meta 中提取 rating
    rating = safe_float(meta.get("rating"), None)

    # If necessary fields are missing, mark as unknown  # 如果缺少必要字段，标记为 unknown
    if fact_label == "fake":
        return "fake"

    # Fake heuristic  # 启发式假评论检测
    is_fake = fake_score >= fake_threshold or base_label == "fake"

    if is_fake:
        return "fake"

    # Decide whether we want to call it real  # 决定标记为 real
    if len(ctx) >= 60 and rating is not None:
        return "real"

    return "fake"  # Default to fake if nothing matches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, default="data/reviews_context.jsonl")
    parser.add_argument("--out_file", type=str, default="data/reviews_with_labels.jsonl")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    retriever = Retriever()  # 初始化检索器

    # Pre-count total lines for tqdm  # 预先统计总行数用于 tqdm
    with open(args.in_file, "r", encoding="utf-8") as fin:
        total_lines = sum(1 for _ in fin)

    total = 0
    real_count = 0
    fake_count = 0

    with open(args.in_file, "r", encoding="utf-8") as fin, open(
            args.out_file, "w", encoding="utf-8"
    ) as fout:
        pbar = tqdm(fin, total=total_lines, desc="Processing reviews")

        for line in pbar:
            j = json.loads(line)
            context_text = j.get("context", "")
            base_label = j.get("pseudo_label", None)

            # 1) RAG factual consistency  # 1) RAG 事实一致性检测
            retrieved = retriever.retrieve(context_text, top_k=args.top_k)
            fact_label = detect_factual_mismatch(context_text, retrieved)

            # 2) Fake heuristic  # 2) 假评论启发式打分
            fake_score = compute_fake_score(j)

            # 3) Final label  # 3) 最终伪标签
            final_label = decide_final_label(fact_label, fake_score, j)

            if final_label == "fake":
                fake_count += 1
            else:
                real_count += 1

            j["retrieved"] = retrieved
            j["factual_consistency"] = fact_label
            j["fake_score"] = fake_score
            j["pseudo_label_v2"] = final_label

            fout.write(json.dumps(j, ensure_ascii=False) + "\n")
            total += 1

            pbar.set_postfix(real=real_count, fake=fake_count, total=total)

    print("[INFO] finished:", total)
    print("[STATS] real:", real_count, "fake:", fake_count, "total:", total)
