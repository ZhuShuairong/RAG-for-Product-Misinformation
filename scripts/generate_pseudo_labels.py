#!/usr/bin/env python3
"""
generate_pseudo_labels.py
对 reviews_context.jsonl 中的每条记录做检索并基于简单规则打标签（事实性矛盾检测）。
Usage:
    python scripts/generate_pseudo_labels.py --in_file data/reviews_context.jsonl --out_file data/reviews_with_labels.jsonl --top_k 3
"""

import argparse
import json
from build_retriever import Retriever
from tqdm import tqdm


def detect_factual_mismatch(review_context, retrieved_docs):
    """Detect simple factual mismatches between review text and retrieved product docs.  # 检测评论文本与检索到的商品文档之间的简单事实性矛盾
    """
    text = review_context.lower()
    for doc in retrieved_docs:
        doc_text = doc["document"].lower()
        # check for fragrance contradiction: review mentions smell/fragrance but product is fragrance-free  # 检查香味矛盾：评论提到气味/香味但商品是无香料
        if ("smell" in text or "fragrance" in text) and "fragrance-free" in doc_text:
            return "factual_error"
        # review mentions perfume/fragrance/smell and product text confirms fragrance (no fragrance-free)  # 评论提到香水/香味/味道且商品文案确实包含香味（且不是无香料）
        if (
                ("fragrance" in text or "perfume" in text or "smell" in text)
                and "fragrance" in doc_text
                and "fragrance-free" not in doc_text
        ):
            return "consistent"
    # if we cannot detect any simple rule-based signal, mark as unknown  # 如果没有命中任何简单规则信号，则标记为 unknown
    return "unknown"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in_file", type=str, default="data/reviews_context.jsonl")
    p.add_argument("--out_file", type=str, default="data/reviews_with_labels.jsonl")
    p.add_argument("--top_k", type=int, default=3)
    args = p.parse_args()

    # Initialize retriever  # 初始化检索器
    retriever = Retriever()

    # Pre-count total lines for tqdm progress bar  # 预先统计总行数以便 tqdm 显示总进度
    with open(args.in_file, "r", encoding="utf-8") as fin:
        total_lines = sum(1 for _ in fin)

    total = 0  # total processed rows  # 已处理的总行数
    # Counters for different label types  # 不同标签类型的计数器
    real_count = 0  # final real-like samples  # 最终视为真实(real/consistent)的样本数
    fake_count = 0  # heuristic fake samples  # 启发式(fake)的样本数
    factual_error_count = 0  # factual_error samples  # factual_error 样本数

    with open(args.in_file, "r", encoding="utf-8") as fin, open(
            args.out_file, "w", encoding="utf-8"
    ) as fout:
        # Initialize tqdm progress bar  # 初始化 tqdm 进度条
        pbar = tqdm(fin, desc="Processing reviews", total=total_lines)

        # Iterate with tqdm to show progress  # 使用 tqdm 迭代文件并显示进度条
        for line in pbar:
            j = json.loads(line)
            context_text = j.get("context", "")
            base_label = j.get("pseudo_label", None)  # original heuristic label  # 原始启发式标签

            # Retrieve related product documents  # 检索相关商品文档
            res = retriever.retrieve(context_text, top_k=args.top_k)

            # Apply factual mismatch rule  # 应用事实矛盾规则
            new_label = detect_factual_mismatch(context_text, res)
            j["retrieved"] = res
            j["pseudo_label_v2"] = new_label

            # --- statistics logic  # --- 统计逻辑部分
            # We aggregate counts with the following intuition:  # 聚合计数的直观逻辑：
            # - factual_error: detected contradiction  # - factual_error：检测到明显矛盾
            # - real: either consistent by retrieval, or originally tagged as real  # - real：检索结果一致(consistent)或最初启发式为 real
            # - fake: originally tagged as fake (not factual_error)  # - fake：启发式为 fake（但未被规则标为 factual_error）
            if new_label == "factual_error":
                factual_error_count += 1
            elif new_label == "consistent":
                real_count += 1
            else:
                # If retrieval cannot decide, fall back to heuristic label  # 如果检索无法决定，则回退到启发式标签
                if base_label == "fake":
                    fake_count += 1
                elif base_label == "real":
                    real_count += 1
                # other values such as "suspicious" or None are ignored here  # 其他如 "suspicious"/None 在这里不计入三类统计

            # Write out augmented record  # 写出带有新标签的记录
            fout.write(json.dumps(j, ensure_ascii=False) + "\n")
            total += 1

            # Update progress bar with stats  # 在进度条上实时更新统计信息
            pbar.set_postfix(
                real=real_count, fake=fake_count, factual_error=factual_error_count, total=total
            )  # 在进度条上显示 real/fake/factual_error/total 的实时数量

    print("[INFO] finished:", total)
    print("[STATS] real:", real_count, "fake:", fake_count, "factual_error:", factual_error_count)
