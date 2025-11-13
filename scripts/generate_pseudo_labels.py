import argparse
import json
from build_retriever import Retriever
from tqdm import tqdm

def detect_factual_mismatch(review_context, retrieved_docs):
    # 检测评论中的事实性矛盾（例如香味问题）
    text = review_context.lower()
    for doc in retrieved_docs:
        doc_text = doc['document'].lower()
        # check for fragrance contradiction:
        if ("smell" in text or "fragrance" in text) and "fragrance-free" in doc_text:
            return "factual_error"
        if ("fragrance" in text or "perfume" in text or "smell" in text) and "fragrance" in doc_text and "fragrance-free" not in doc_text:
            # consistent with fragrance
            return "consistent"
    return "unknown"

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in_file", type=str, default="data/reviews_context.jsonl")
    p.add_argument("--out_file", type=str, default="data/reviews_with_labels.jsonl")
    p.add_argument("--top_k", type=int, default=3)
    args = p.parse_args()

    retriever = Retriever()

    # 计算文件的总行数
    with open(args.in_file, "r", encoding="utf-8") as fin:
        total_lines = sum(1 for _ in fin)

    total = 0
    with open(args.in_file, "r", encoding="utf-8") as fin, open(args.out_file, "w", encoding="utf-8") as fout:
        # 使用 tqdm 显示进度条
        for line in tqdm(fin, desc="Processing reviews", total=total_lines):
            j = json.loads(line)
            context_text = j.get("context","")
            res = retriever.retrieve(context_text, top_k=args.top_k)
            label = detect_factual_mismatch(context_text, res)
            j["retrieved"] = res
            j["pseudo_label_v2"] = label
            fout.write(json.dumps(j, ensure_ascii=False) + "\n")
            total += 1
    print("[INFO] finished:", total)
