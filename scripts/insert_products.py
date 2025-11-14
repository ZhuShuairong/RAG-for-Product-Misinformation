import argparse
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm
import numpy as np


def create_product_text(row):
    """创建完整的产品描述文本，包含所有关键字段"""
    parts = []

    # 产品基本信息
    name = row.get('product_name')
    brand = row.get('brand_name')
    if pd.notna(name):
        parts.append(f"Product Name: {name}")
    if pd.notna(row.get('product_id', None)):
        parts.append(f"Product ID: {str(row.get('product_id'))}")
    if pd.notna(brand):
        parts.append(f"Brand: {brand}")
    if pd.notna(row.get('brand_id', None)):
        parts.append(f"Brand ID: {str(row.get('brand_id'))}")

    # 产品详细信息
    if pd.notna(row.get('loves_count', None)):
        parts.append(f"Loves Count: {str(row.get('loves_count'))}")
    if pd.notna(row.get('rating', None)):
        parts.append(f"Rating: {str(row.get('rating'))}")
    if pd.notna(row.get('reviews', None)):
        parts.append(f"Number of Reviews: {str(row.get('reviews'))}")
    if pd.notna(row.get('size', None)):
        parts.append(f"Size: {str(row.get('size'))}")
    if pd.notna(row.get('variation_type', None)):
        parts.append(f"Variation Type: {str(row.get('variation_type'))}")
    if pd.notna(row.get('variation_value', None)):
        parts.append(f"Variation Value: {str(row.get('variation_value'))}")
    if pd.notna(row.get('variation_desc', None)):
        parts.append(f"Variation Description: {str(row.get('variation_desc'))}")
    if pd.notna(row.get('ingredients', None)):
        parts.append(f"Ingredients: {str(row.get('ingredients'))}")
    if pd.notna(row.get('price_usd', None)):
        parts.append(f"Price: ${str(row.get('price_usd'))}")
    if pd.notna(row.get('value_price_usd', None)):
        parts.append(f"Value Price: ${str(row.get('value_price_usd'))}")
    if pd.notna(row.get('sale_price_usd', None)):
        parts.append(f"Sale Price: ${str(row.get('sale_price_usd'))}")
    if pd.notna(row.get('limited_edition', None)):
        parts.append(f"Limited Edition: {str(row.get('limited_edition'))}")
    if pd.notna(row.get('new', None)):
        parts.append(f"New: {str(row.get('new'))}")
    if pd.notna(row.get('online_only', None)):
        parts.append(f"Online Only: {str(row.get('online_only'))}")
    if pd.notna(row.get('out_of_stock', None)):
        parts.append(f"Out of Stock: {str(row.get('out_of_stock'))}")
    if pd.notna(row.get('sephora_exclusive', None)):
        parts.append(f"Sephora Exclusive: {str(row.get('sephora_exclusive'))}")
    if pd.notna(row.get('highlights', None)):
        parts.append(f"Highlights: {str(row.get('highlights'))}")
    if pd.notna(row.get('primary_category', None)):
        parts.append(f"Primary Category: {str(row.get('primary_category'))}")
    if pd.notna(row.get('secondary_category', None)):
        parts.append(f"Secondary Category: {str(row.get('secondary_category'))}")
    if pd.notna(row.get('tertiary_category', None)):
        parts.append(f"Tertiary Category: {str(row.get('tertiary_category'))}")
    if pd.notna(row.get('child_count', None)):
        parts.append(f"Child Count: {str(row.get('child_count'))}")
    if pd.notna(row.get('child_max_price', None)):
        parts.append(f"Child Max Price: ${str(row.get('child_max_price'))}")
    if pd.notna(row.get('child_min_price', None)):
        parts.append(f"Child Min Price: ${str(row.get('child_min_price'))}")

    return ". ".join(parts)  # 生成完整的产品描述文本


def insert_products_chroma(products_csv, persist_dir="chroma_db", batch_size=256, model_name="all-MiniLM-L6-v2"):
    """将产品信息插入到 Chroma DB"""
    print("[INFO] loading products:", products_csv)
    df = pd.read_csv(products_csv, dtype=str)
    df = df.fillna("")  # 处理缺失值，确保不会出错

    # 初始化 Chroma 客户端
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(name="products")

    # 初始化 SentenceTransformer 模型
    embedder = SentenceTransformer(model_name)

    # 批量插入
    docs, metadatas, ids = [], [], []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Inserting products"):
        pid = str(row.get("product_id") or f"prod_{idx}")

        # 使用 create_product_text 函数生成详细的产品描述
        product_description = create_product_text(row)

        # 将产品的所有字段直接存入 meta
        product_meta = {
            "product_id": pid,
            "product_name": row.get("product_name", ""),
            "brand_id": row.get("brand_id", ""),
            "brand_name": row.get("brand_name", ""),
            "loves_count": row.get("love_count", 0),
            "rating": row.get("rating", ""),
            "reviews": row.get("reviews", 0),
            "size": row.get("size", ""),
            "variation_type": row.get("variation_type", ""),
            "variation_value": row.get("variation_value", ""),
            "variation_desc": row.get("variation_desc", ""),
            "ingredients": row.get("ingredients", ""),
            "price_usd": row.get("price_usd", 0),
            "value_price_usd": row.get("value_price_usd") if row.get("value_price_usd") else row.get("price_usd", 0),
            "sale_price_usd": row.get("sale_price_usd") if row.get("sale_price_usd") else row.get("price_usd", 0),
            "limited_edition": row.get("limited_edition", 0),
            "new": row.get("new", 0),
            "online_only": row.get("online_only", 0),
            "out_of_stock": row.get("out_of_stock", 0),
            "sephora_exclusive": row.get("sephora_exclusive", 0),
            "highlights": row.get("highlights", ""),
            "primary_category": row.get("primary_category", ""),
            "secondary_category": row.get("secondary_category", ""),
            "tertiary_category": row.get("tertiary_category", ""),
            "child_count": row.get("child_count", 0),
            "child_max_price": row.get("child_max_price", 0),
            "child_min_price": row.get("child_min_price", 0)
        }

        # 将生成的描述作为 "documents"，产品元数据作为 "meta"
        docs.append(product_description)  # 存储详细描述
        metadatas.append(product_meta)  # 存储所有元数据
        ids.append(pid)  # 使用产品 ID 作为唯一标识符

        # 批量处理嵌入
        if len(docs) >= batch_size:
            emb = embedder.encode(docs, convert_to_numpy=True, show_progress_bar=False)
            collection.add(
                documents=docs,
                metadatas=metadatas,
                ids=ids,
                embeddings=emb.tolist()
            )
            docs, metadatas, ids = [], [], []  # 重置批次

    # 处理剩余的数据
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
    p.add_argument("--products_csv", type=str, required=True, help="CSV file containing product information", default="data/product_info.csv")
    p.add_argument("--persist_dir", type=str, required=True, help="Directory to persist Chroma DB", default="chroma_db")
    p.add_argument("--batch_size", type=int, default=256, help="Batch size for embeddings")
    p.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2", help="Sentence transformer model name")
    args = p.parse_args()
    insert_products_chroma(args.products_csv, args.persist_dir, args.batch_size, args.model_name)
