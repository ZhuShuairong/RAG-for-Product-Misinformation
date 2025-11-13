#!/usr/bin/env python3
"""
prepare_context.py
从 reviews CSV 生成 streaming JSONL，每条包含 context 信息与 pseudo-label
Usage:
    python scripts/prepare_context.py --input_dir data --out_file data/reviews_context.jsonl --max_rows 100000
"""
import argparse
import os
import glob
import json
import pandas as pd
from tqdm import tqdm
import math
import re

def find_review_files(input_dir: str):
    files = sorted(glob.glob(os.path.join(input_dir, "reviews*.csv")))
    if not files:
        # fallback to any csv
        files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    return files

def heuristic_label(row):
    # 你可以根据需要扩展或替换为更复杂规则
    text = str(row.get('review_text', '')).strip()
    rating = None
    try:
        rating = float(row.get('rating', 0))
    except:
        rating = None
    helpfulness = 0.0
    try:
        helpfulness = float(row.get('helpfulness', 0))
    except:
        helpfulness = 0.0

    lower = text.lower()
    # short + extreme positive + low helpfulness -> fake
    if len(lower) < 20 and rating == 5 and helpfulness < 0.2:
        return "fake"
    # repetitive promotional words
    promo_patterns = ["must buy", "best ever", "life changing", "buy now", "i love this product", "highly recommend"]
    if any(p in lower for p in promo_patterns) and len(lower) < 50 and rating >= 4:
        return "suspicious"
    # otherwise label as unknown/real by default
    return "real"

def build_context_text(row, product_lookup=None):
    # product_lookup: optional dict product_id -> product_text
    parts = []
    rt = row.get('review_text', '')
    if pd.notna(row.get('review_title', None)):
        parts.append(f"Review title: {row['review_title']}")
    parts.append(f"Review text: {rt}")
    if pd.notna(row.get('rating', None)):
        parts.append(f"Rating: {row['rating']}")
    if pd.notna(row.get('is_recommended', None)):
        parts.append(f"Recommended: {row['is_recommended']}")
    # attach short product summary if available
    pid = row.get('product_id')
    if product_lookup and pid in product_lookup:
        parts.append("Product summary: " + product_lookup[pid])
    return " | ".join([p for p in parts if p])

def process_files(files, out_file, product_lookup=None, max_rows=None):
    total_written = 0
    with open(out_file, "w", encoding="utf-8") as fout:
        for fpath in files:
            print(f"[INFO] reading {fpath}")
            # read in chunks to avoid OOM
            for chunk in pd.read_csv(fpath, chunksize=10000, iterator=True, encoding='utf-8', dtype=str):
                # convert columns and fillna minimal
                for _, row in chunk.iterrows():
                    if max_rows is not None and total_written >= max_rows:
                        print("[INFO] reached max_rows limit.")
                        return total_written
                    rowd = row.to_dict()
                    context_text = build_context_text(rowd, product_lookup)
                    label = heuristic_label(rowd)
                    out_entry = {
                        "review_id": rowd.get("review_id") or None,
                        "product_id": rowd.get("product_id"),
                        "context": context_text,
                        "meta": {
                            "rating": rowd.get("rating"),
                            "is_recommended": rowd.get("is_recommended"),
                            "helpfulness": rowd.get("helpfulness"),
                            "submission_time": rowd.get("submission_time")
                        },
                        "pseudo_label": label
                    }
                    fout.write(json.dumps(out_entry, ensure_ascii=False) + "\n")
                    total_written += 1
    print(f"[INFO] completed writing {total_written} entries to {out_file}")
    return total_written

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, default="data")
    p.add_argument("--out_file", type=str, default="data/reviews_context.jsonl")
    p.add_argument("--products_csv", type=str, default="data/products.csv")
    p.add_argument("--max_rows", type=int, default=None)
    args = p.parse_args()

    files = find_review_files(args.input_dir)
    print("[INFO] found files:", files[:5])
    product_lookup = {}
    if os.path.exists(args.products_csv):
        dfp = pd.read_csv(args.products_csv, dtype=str)
        for _, r in dfp.iterrows():
            pid = r.get("product_id")
            # create short product summary
            prod_summary = f"{r.get('product_name','')} by {r.get('brand_name','')}. Highlights: {r.get('highlights','')}. Ingredients: {r.get('ingredients','')}"
            product_lookup[pid] = prod_summary
        print(f"[INFO] loaded {len(product_lookup)} products for join")
    process_files(files, args.out_file, product_lookup, args.max_rows)
