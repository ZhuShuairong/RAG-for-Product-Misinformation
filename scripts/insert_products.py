#!/usr/bin/env python3
"""
insert_products.py (Chroma new API compatible version)
适配 Chroma >= 0.5.0
Usage:
    python scripts/insert_products.py --products_csv data/products.csv --persist_dir chroma_db --batch_size 256
"""
import argparse
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm
import numpy as np


def create_product_text(row):
    parts = []
    name = row.get('product_name')
    brand = row.get('brand_name')
    if pd.notna(name):
        parts.append(f"{name}")
    if pd.notna(brand):
        parts.append(f"by {brand}")
    if pd.notna(row.get('highlights', None)):
        parts.append("Highlights: " + str(row.get('highlights')))
    if pd.notna(row.get('ingredients', None)):
        parts.append("Ingredients: " + str(row.get('ingredients')))
    if pd.notna(row.get('primary_category', None)):
        parts.append("Category: " + str(row.get('primary_category')))
    if pd.notna(row.get('price_usd', None)):
        parts.append("Price: $" + str(row.get('price_usd')))
    if pd.notna(row.get('rating', None)):
        parts.append("Rating: " + str(row.get('rating')))
    return ". ".join(parts)


def insert_products_chroma(products_csv, persist_dir="chroma_db", batch_size=256, model_name="all-MiniLM-L6-v2"):
    print("[INFO] loading products:", products_csv)
    df = pd.read_csv(products_csv, dtype=str)
    df = df.fillna("")

    # 初始化向量模型
    embedder = SentenceTransformer(model_name)

    # ✅ 新版 Chroma 初始化方式
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(name="products")

    # 批量插入
    docs, metadatas, ids = [], [], []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Inserting products"):
        pid = str(row.get("product_id") or f"prod_{idx}")
        text = create_product_text(row)
        meta = {
            "product_id": pid,
            "product_name": row.get("product_name", ""),
            "brand_name": row.get("brand_name", ""),
            "price_usd": row.get("price_usd", ""),
            "rating": row.get("rating", "")
        }
        docs.append(text)
        metadatas.append(meta)
        ids.append(pid)

        if len(docs) >= batch_size:
            emb = embedder.encode(docs, convert_to_numpy=True, show_progress_bar=False)
            collection.add(
                documents=docs,
                metadatas=metadatas,
                ids=ids,
                embeddings=emb.tolist()
            )
            docs, metadatas, ids = [], [], []

    # 处理剩余
    if docs:
        emb = embedder.encode(docs, convert_to_numpy=True, show_progress_bar=False)
        collection.add(
            documents=docs,
            metadatas=metadatas,
            ids=ids,
            embeddings=emb.tolist()
        )

    print(f"[INFO] ✅ inserted {len(df)} products into Chroma at {persist_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--products_csv", type=str, default="data/product_info.csv")
    p.add_argument("--persist_dir", type=str, default="chroma_db")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2")
    args = p.parse_args()
    insert_products_chroma(args.products_csv, args.persist_dir, args.batch_size, args.model_name)
