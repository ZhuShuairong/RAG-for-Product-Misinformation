# Phase 2 - Feature Engineering & RAG Preparation
# 第 2 阶段 - 特征工程和 RAG 准备
# This phase prepares features for the RAG (Retrieval-Augmented Generation) system. It creates comprehensive product profiles, embeds them using Sentence Transformers, stores them in a ChromaDB vector database for efficient retrieval, generates training examples with explanations for fake review detection, and saves them for model fine-tuning.
# 该阶段为 RAG（检索增强生成）系统准备特征。它创建了全面的产品简介，使用 Sentence Transformers 对其进行嵌入，将其存储在 ChromaDB 向量数据库中以实现高效检索，生成带有解释的假评论检测训练示例，并将其保存以进行模型微调。
# Depends on synthetic_train.csv from phase 1, and product_info_clean from phase 0.
# 依赖于第 1 阶段的 synthetic_train.csv 和第 0 阶段的 product_info_clean。

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# Load synthetic datasets  # 加载合成数据集
train_df = pd.read_csv('./data/synthetic_train.csv')
test_df = pd.read_csv('./data/synthetic_test.csv')

print(f"Loaded training: {len(train_df)}, testing: {len(test_df)} samples")

# Load cleaned product_info from phase 1  # 从第1阶段加载清理后的product_info
product_info_clean = pd.read_csv('./data/product_info_clean.csv')
print("Loaded cleaned product_info")

# Create product profiles  # 创建产品简介
product_profiles = []
product_ids = []
product_metadatas = []

for idx, row in product_info_clean.iterrows():
    profile = f"""
Product: {row['product_name']}
Brand: {row['brand_name']}
Category: {row['primary_category']}
Price: ${row['price_usd']:.2f}
Ingredients: {row['ingredients'] if pd.notna(row['ingredients']) else 'Not Listed'}
Highlights: {row['highlights'] if pd.notna(row['highlights']) else 'None'}
""".strip()

    product_profiles.append(profile)
    product_ids.append(str(row['product_id']))
    product_metadatas.append({
        "product_name": row['product_name'],
        "brand_name": row['brand_name'],
        "category": row['primary_category'],
        "price": float(row['price_usd'])
    })

print(f"Created {len(product_profiles)} product profiles")

# Setup ChromaDB  # 设置ChromaDB
client = chromadb.PersistentClient(path="./data/chroma_data")
product_profile_collection = client.get_or_create_collection(
    name="product_profiles",
    metadata={"hnsw:space": "cosine"}
)

# Embed and store product profiles  # 嵌入并存储产品简介
if product_profile_collection.count() == 0:
    # Embed and store product profiles  # 如果集合为空，则嵌入并存储产品简介
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Generating embeddings...")
    profile_embeddings = embedding_model.encode(product_profiles, show_progress_bar=True)

    batch_size = 5000
    print(f"Storing in batches of {batch_size}...")
    for i in range(0, len(product_profiles), batch_size):
        batch_end = min(i + batch_size, len(product_profiles))
        batch_ids = product_ids[i:batch_end]
        batch_profiles = product_profiles[i:batch_end]
        batch_embeddings = profile_embeddings[i:batch_end].tolist()
        batch_metadatas = product_metadatas[i:batch_end]

        product_profile_collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_profiles,
            metadatas=batch_metadatas
        )
        print(f"Added {batch_end}/{len(product_profiles)} profiles")
    print("Vector database initialized.")
else:
    print("Vector database already exists.")

# Test retrieval  # 测试检索
test_profile = product_profile_collection.query(
    query_texts=["A hydrating moisturizer for dry skin"],
    n_results=2
)
print("Test retrieval completed.")

# Check if training examples already exist  # 检查训练示例是否已存在
import os

if os.path.exists('./data/training_examples.json'):
    print("Training examples already exist, skipping creation.")
    # Load existing for verification
    import json

    with open('./data/training_examples.json', 'r') as f:
        training_examples = json.load(f)
    print(f"Loaded {len(training_examples)} existing training examples")
else:
    print("Training examples not found, proceeding with creation...")

# Prepare training data for examples  # 准备训练数据以生成示例
train_df['review_text'] = train_df['review_text'].fillna('').astype(str)
train_df['product_info'] = train_df['product_info'].fillna('').astype(str)


# Define heuristic explanation generation function for training examples  # 定义用于训练示例的启发式解释生成函数
def generate_heuristic_explanation(row):
    # Generate explanation for fake reviews based on simple heuristics  # 为假评论基于简单启发式生成解释
    is_fake = int(row.get('is_fake', 0))
    if is_fake == 0:
        return "This review matches the product information and appears authentic."
    else:
        review_text = str(row.get('review_text', '')).lower()
        product_info = str(row.get('product_info', '')).lower()
        explanations = []
        if 'skincare' in review_text and 'hair' in product_info:
            explanations.append("Review discusses skincare but product is for hair.")
        elif 'hair' in review_text and 'skincare' in product_info:
            explanations.append("Review discusses hair but product is for skincare.")
        if 'natural' in review_text and 'chemical' in product_info:
            explanations.append("Review praises natural ingredients but product contains chemicals.")
        if 'drying' in review_text and 'hydrating' in product_info:
            explanations.append("Review mentions drying effects but product is hydrating.")
        elif 'hydrating' in review_text and 'drying' in product_info:
            explanations.append("Review mentions hydrating effects but product may be drying.")
        if not explanations:
            explanations.append("Review appears mismatched with product information.")
        return " ".join(explanations)


# Create training examples  # 创建训练示例
training_examples = []
for idx, row in train_df.iterrows():
    # Generate an explanation using the heuristic generator defined earlier.  # 使用前面定义的启发式生成器生成解释。
    # Fall back to a short default string if the heuristic fails for any row.  # 如果启发式方法对任何行失败，则回退到简短的默认字符串。
    try:
        explanation = generate_heuristic_explanation(row)
    except Exception as e:
        explanation = "No explanation available (error generating heuristic): " + str(e)

    example = {
        "product_info": row['product_info'],
        "review_text": row['review_text'],
        "label": int(row['is_fake']),
        "explanation_template": explanation,
    }
    training_examples.append(example)

print(f"Created {len(training_examples)} training examples")

# Save training examples  # 保存训练示例
import json

with open('./data/training_examples.json', 'w') as f:
    json.dump(training_examples, f, indent=2)

print("Training examples saved.")

# Show sample training examples  # 显示示例训练示例
for ex in training_examples[:2]:
    print(f"Label: {ex['label']}, Explanation: {ex['explanation_template'][:50]}...")

print("Phase 2 completed: Product profiles in ChromaDB, training examples prepared.")

# This phase creates files: chroma_data (ChromaDB database) and training_examples.json in the data folder.
# 该阶段创建文件：data文件夹中的chroma_data（ChromaDB数据库）和training_examples.json。
