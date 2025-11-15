import argparse
import os
import glob
import json
import pandas as pd
from tqdm import tqdm


def find_review_files(input_dir: str):
    """查找并返回所有评论 CSV 文件"""
    files = sorted(glob.glob(os.path.join(input_dir, "reviews*.csv")))
    if not files:
        # 如果没有找到匹配的文件，回退到所有 CSV 文件
        files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    return files


def build_context_text(row, product_lookup=None):
    """构建评论的上下文文本"""
    parts = []
    rt = row.get('review_text', '')
    if pd.notna(row.get('review_title', None)):
        parts.append(f"Review title: {row['review_title']}")
    parts.append(f"Review text: {rt}")
    if pd.notna(row.get('rating', None)):
        parts.append(f"Rating: {row['rating']}")
    if pd.notna(row.get('is_recommended', None)):
        parts.append(f"Recommended: {row['is_recommended']}")
    # 如果有产品摘要信息，添加产品摘要
    pid = row.get('product_id')
    if product_lookup and pid in product_lookup:
        parts.append("Product summary: " + product_lookup[pid])
    return " | ".join([p for p in parts if p])


def process_files(files, out_file, product_lookup=None, max_rows=None):
    """处理所有文件并生成 JSONL 格式的数据"""
    total_written = 0
    with open(out_file, "w", encoding="utf-8") as fout:
        for fpath in files:
            print(f"[INFO] reading {fpath}")
            # 逐块读取 CSV 文件以避免内存溢出
            for chunk in pd.read_csv(fpath, chunksize=10000, iterator=True, encoding='utf-8', dtype=str, low_memory=False):
                # 转换并处理数据
                for _, row in chunk.iterrows():
                    if max_rows is not None and total_written >= max_rows:
                        print("[INFO] reached max_rows limit.")
                        return total_written
                    rowd = row.to_dict()
                    context_text = build_context_text(rowd, product_lookup)
                    out_entry = {
                        "review_id": rowd.get("review_id") or None,
                        "product_id": rowd.get("product_id"),
                        "context": context_text,
                        "meta": {
                            "author_id": rowd.get("author_id", None),
                            "rating": rowd.get("rating", None),
                            "is_recommended": rowd.get("is_recommended", None),
                            "helpfulness": rowd.get("helpfulness", None),
                            "total_feedback_count": rowd.get("total_feedback_count", None),
                            "total_neg_feedback_count": rowd.get("total_neg_feedback_count", None),
                            "total_pos_feedback_count": rowd.get("total_pos_feedback_count", None),
                            # "submission_time": rowd.get("submission_time", None),
                            # "review_text": rowd.get("review_text", None),
                            # "review_title": rowd.get("review_title", None),
                            # "skin_tone": rowd.get("skin_tone", None),
                            # "eye_color": rowd.get("eye_color", None),
                            # "skin_type": rowd.get("skin_type", None),
                            # "hair_color": rowd.get("hair_color", None),
                            # "product_id": rowd.get("product_id", None)
                        },
                        # 初始标签为 None，后续会在 generate_pseudo_labels.py 中标注
                        "pseudo_label": None
                    }
                    fout.write(json.dumps(out_entry, ensure_ascii=False) + "\n")
                    total_written += 1
    print(f"[INFO] completed writing {total_written} entries to {out_file}")
    return total_written


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, default="data")
    p.add_argument("--out_file", type=str, default="data/reviews_context.jsonl")
    p.add_argument("--products_csv", type=str, default="data/product_info.csv")
    p.add_argument("--max_rows", type=int, default=None)
    args = p.parse_args()

    files = find_review_files(args.input_dir)
    print("[INFO] found files:", files[:5])
    product_lookup = {}
    if os.path.exists(args.products_csv):
        dfp = pd.read_csv(args.products_csv, dtype=str, low_memory=False)
        for _, r in dfp.iterrows():
            pid = r.get("product_id")
            # 创建简短的产品摘要
            prod_summary = f"{r.get('product_name', '')} by {r.get('brand_name', '')}. Highlights: {r.get('highlights', '')}."
            product_lookup[pid] = prod_summary
        print(f"[INFO] loaded {len(product_lookup)} products for join")
    process_files(files, args.out_file, product_lookup, args.max_rows)
