# Phase 1 - Synthetic Dataset Creation for Fake Review Detection
# 第 1 阶段 - 用于假评论检测的合成数据集创建
# This phase creates synthetic datasets for training a fake review detection model. It merges cleaned reviews with product information, generates fake reviews by shuffling real review texts, combines real and fake data, and splits into training and testing sets with stratification to ensure balanced labels.
# 此阶段创建用于训练虚假评论检测模型的合成数据集。它将清洗后的评论与产品信息合并，通过打乱真实评论文本生成虚假评论，将真实数据和虚假数据混合，并采用分层抽样方法将数据集拆分为训练集和测试集，以确保标签平衡。
# Depends on cleaned dataframes product_info_clean and reviews_clean from phase 0.
# 取决于第 0 阶段的已清理数据框 product_info_clean 和 reviews_clean。

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# Enhanced Synthetic Data Generation Function (Safer & Faster)  # 增强的合成数据生成函数（更安全且更快）
def generate_synthetic_fakes_with_product_swap(merged_df, fake_ratio=0.2, max_rows=None):
    """
    Generate fake reviews by swapping reviews across different product categories while keeping product labels the same.
    通过交换不同产品类别下的评论，同时保持产品标签不变，生成虚假评论。

    This safer version:
    此更安全的版本：
    - Optional `max_rows` to operate on a sampled subset for quick runs.
    - 可选的 `max_rows` 参数，用于对抽样子集进行操作，以加快运行速度。
    - Precomputes category groups and all_reviews once.
    - 预先计算一次类别分组和所有评论。
    - Avoids repeated expensive list constructions.
    - 避免重复进行耗时的列表构建。
    - Adds light progress logging and deterministic randomness.
    - 添加了轻量级的进度日志记录和确定性随机性。
    """
    import numpy as _np
    from tqdm import tqdm

    # Work on a sample if requested to avoid very long runs during interactive debugging  # 如果需要，在交互式调试期间避免非常长的运行时间，可以对样本进行处理
    if max_rows is not None and len(merged_df) > max_rows:
        df = merged_df.sample(n=max_rows, random_state=42).reset_index(drop=True)
        print(f"Using sampled subset of {max_rows} rows for synthetic generation (from {len(merged_df)} total rows)")
    else:
        df = merged_df.reset_index(drop=True)

    _np.random.seed(42)

    # Pre-group by category for efficient sampling  # 通过类别预分组以实现高效采样
    categories = df['primary_category'].unique()
    category_groups = {cat: df[df['primary_category'] == cat]['review_text'].tolist() for cat in categories}

    # Precompute a flat list of all reviews as fallback  # 预先计算所有评论的平面列表作为后备
    all_reviews = [rev for lst in category_groups.values() for rev in lst]
    if not all_reviews:
        # If there are no reviews at all, return empty frame  # 如果根本没有评论，则返回空框架
        print("No reviews available to generate fakes from.")
        df['is_fake'] = 0
        return df

    num_fake = max(1, int(fake_ratio * len(df)))
    fake_indices = _np.random.choice(len(df), num_fake, replace=False)

    print(f"Generating {num_fake} fake reviews...")

    # VECTORIZED APPROACH: Precompute categories for all fake indices  # 向量化方法：预计算所有虚假索引的类别
    fake_categories = df.loc[fake_indices, 'primary_category'].values
    other_categories_list = []
    available_reviews_list = []

    for cat in tqdm(fake_categories, desc="Processing fake categories"):
        other_cats = [c for c in categories if c != cat]
        if other_cats:
            random_cat = _np.random.choice(other_cats)
            reviews = category_groups.get(random_cat, [])
            # LIMIT REVIEW LIST SIZE FOR EFFICIENCY - sample at most 1000 reviews per category  # 为了效率限制评论列表大小 - 每个类别最多采样1000条评论
            if len(reviews) > 1000:
                reviews = _np.random.choice(reviews, size=1000, replace=False).tolist()
            if reviews:
                available_reviews_list.append(reviews)
                other_categories_list.append(random_cat)
            else:
                # Limit fallback list size too  # 限制后备列表大小
                limited_all_reviews = all_reviews if len(all_reviews) <= 1000 else _np.random.choice(all_reviews, size=1000, replace=False).tolist()
                available_reviews_list.append(limited_all_reviews)
                other_categories_list.append('fallback')
        else:
            # Limit fallback list size too  # 限制后备列表大小
            limited_all_reviews = all_reviews if len(all_reviews) <= 1000 else _np.random.choice(all_reviews, size=1000, replace=False).tolist()
            available_reviews_list.append(limited_all_reviews)
            other_categories_list.append('fallback')

    # Now generate the fake reviews efficiently - VECTORIZED APPROACH  # 现在高效地生成虚假评论 - 向量化方法
    print(f"Selecting {len(available_reviews_list)} fake reviews...")

    # VECTORIZED RANDOM SELECTION: Pre-compute all random indices at once for speed  # 向量化随机选择：一次预先计算所有随机索引以提高速度
    all_reviews_lengths = [len(reviews) for reviews in available_reviews_list]
    random_indices = _np.random.randint(0, _np.array(all_reviews_lengths))

    new_reviews = []
    for i, reviews in enumerate(tqdm(available_reviews_list, desc="Selecting fake reviews")):
        if len(reviews) > 0:
            new_reviews.append(reviews[random_indices[i]])
        else:
            # Fallback to all_reviews if category is empty  # 如果类别为空，则回退到所有评论
            fallback_idx = _np.random.randint(0, len(all_reviews))
            new_reviews.append(all_reviews[fallback_idx])

    fake_df = df.iloc[fake_indices].copy()
    fake_df['review_text'] = new_reviews
    fake_df['is_fake'] = 1

    real_df = df.drop(index=fake_indices).copy()
    real_df['is_fake'] = 0

    # Return combined dataset (shuffled)  # 返回合并的数据集（已洗牌）
    combined = pd.concat([real_df, fake_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    return combined


# Load and clean data (similar to Phase 0)  # 加载和清理数据（类似于第0阶段）
print("Loading and cleaning data...")

# Load product_info  # 加载 product_info
try:
    product_info = pd.read_csv('./data/sephora_data/product_info.csv')
    print("Product info loaded successfully")
except FileNotFoundError:
    print("Error: product_info.csv not found.")
    product_info = pd.DataFrame()

# Load reviews  # 加载评论
try:
    review_files = ['./data/sephora_data/reviews_0-250.csv', './data/sephora_data/reviews_250-500.csv', './data/sephora_data/reviews_500-750.csv', './data/sephora_data/reviews_750-1250.csv', './data/sephora_data/reviews_1250-end.csv']
    reviews = pd.concat([pd.read_csv(f, low_memory=False) for f in review_files], ignore_index=True)
    print("Reviews loaded successfully")
except FileNotFoundError:
    print("Error: review files not found.")
    reviews = pd.DataFrame()

# Clean product_info  # 清理 product_info
product_info_clean = product_info.copy()
fill_dict = {
    'rating': product_info_clean['rating'].median(),
    'reviews': product_info_clean['reviews'].median(),
    'size': 'Unknown',
    'variation_type': 'None',
    'variation_value': 'None',
    'variation_desc': 'None',
    'ingredients': 'Not Listed',
    'highlights': 'None',
    'secondary_category': 'Other',
    'tertiary_category': 'Other',
    'child_max_price': 0,
    'child_min_price': 0
}
product_info_clean = product_info_clean.fillna(fill_dict)
product_info_clean['value_price_usd'] = product_info_clean['value_price_usd'].fillna(product_info_clean['price_usd'])
product_info_clean['sale_price_usd'] = product_info_clean['sale_price_usd'].fillna(product_info_clean['price_usd'])

# Save cleaned product_info for use in other phases  # 保存清理后的 product_info 以供其他阶段使用
product_info_clean.to_csv('./data/product_info_clean.csv', index=False)
print("Cleaned product_info saved as 'product_info_clean.csv'")

# Clean reviews  # 清理评论
reviews_clean = reviews.copy()
reviews_clean = reviews_clean.drop('Unnamed: 0', axis=1)
fill_dict_reviews = {
    'is_recommended': 0,
    'helpfulness': 0,
    'review_text': '',
    'review_title': 'No Title',
    'skin_tone': 'Not Specified',
    'eye_color': 'Not Specified',
    'skin_type': 'Not Specified',
    'hair_color': 'Not Specified'
}
reviews_clean = reviews_clean.fillna(fill_dict_reviews)

print("Data loaded and cleaned.")

# Merge reviews with product information  # 将评论与产品信息合并
print(f"Reviews clean shape: {reviews_clean.shape}")
print(f"Product info clean shape: {product_info_clean.shape}")
print(f"Common product_ids: {len(set(reviews_clean['product_id']).intersection(set(product_info_clean['product_id'])))}")

merged_df = pd.merge(reviews_clean, product_info_clean, on='product_id', how='inner', suffixes=('_review', '_product'))
print(f"Merged dataset shape: {merged_df.shape}")
print("Merged columns:", merged_df.columns.tolist()[:10])  # Show first 10


# Create product_info concatenated column  # 创建 product_info 拼接列
def create_product_info(row):
    # Safely handle both suffixed and unsuffixed column names from the merge  # 安全处理合并后的带后缀和不带后缀的列名
    def safe_get(r, *keys, default=None):
        for k in keys:
            if k in r.index and pd.notna(r[k]):
                return r[k]
        return default

    ingredients = safe_get(row, 'ingredients', default='Not Listed')
    highlights = safe_get(row, 'highlights', default='None')
    brand = safe_get(row, 'brand_name_product', 'brand_name', default='Unknown')
    primary_category = safe_get(row, 'primary_category', default='Unknown')
    price = safe_get(row, 'price_usd_product', 'price_usd', default=0.0)

    try:
        price_val = float(price)
    except Exception:
        price_val = 0.0

    return f"Brand: {brand}, Category: {primary_category}, Price: ${price_val:.2f}, Ingredients: {ingredients}, Highlights: {highlights}"


merged_df['product_info'] = merged_df.apply(create_product_info, axis=1)

# Generate synthetic dataset with category-aware mismatches  # 使用类别感知的不匹配生成合成数据集
# Use max_rows=10000 for faster testing, remove for full dataset  # 使用 max_rows=10000 以加快测试速度，完整数据集请移除
combined_df = generate_synthetic_fakes_with_product_swap(merged_df, fake_ratio=0.5, max_rows=10000)

print(f"Real reviews: {len(combined_df[combined_df['is_fake'] == 0])}")
print(f"Fake reviews: {len(combined_df[combined_df['is_fake'] == 1])}")
print(f"Total combined: {len(combined_df)}")

# Select required columns and rename  # 选择所需列并重命名
name_col = None
if 'product_name_product' in combined_df.columns:
    name_col = 'product_name_product'
elif 'product_name' in combined_df.columns:
    name_col = 'product_name'
else:
    # Fallback: create a product_name column if missing  # 后备：如果缺少则创建 product_name 列
    combined_df['product_name'] = combined_df.get('product_name_product', 'Unknown')
    name_col = 'product_name'

cols = ['product_id', name_col, 'review_text', 'is_fake', 'product_info']
final_df = combined_df[cols].copy()
final_df.rename(columns={name_col: 'product_name'}, inplace=True)

# Split into training (5000 samples) and testing (5000 samples) with stratification  # 按照分层划分为训练集（5000个样本）和测试集（5000个样本）
label_counts = final_df['is_fake'].value_counts()
use_stratify = label_counts.min() >= 2
if use_stratify:
    stratify_col = final_df['is_fake']
    print("Using stratified split (each class has >=2 samples)")
else:
    stratify_col = None
    print("Dataset too small or imbalanced for stratified split; using random split without stratification")

train_df, test_df = train_test_split(
    final_df,
    test_size=5000,
    stratify=stratify_col,
    random_state=42
)

print(f"Training set: {len(train_df)} samples")
print(f"Testing set: {len(test_df)} samples")
print(f"Training fake ratio: {train_df['is_fake'].mean():.3f}")
print(f"Testing fake ratio: {test_df['is_fake'].mean():.3f}")

# Save datasets  # 保存数据集
train_df.to_csv('./data/synthetic_train.csv', index=False)
test_df.to_csv('./data/synthetic_test.csv', index=False)

print("✓ Synthetic datasets saved as 'synthetic_train.csv' and 'synthetic_test.csv'")

# Quick verification  # 快速验证
print("\nSample of training data:")
print(train_df.head(2))

# This phase creates files: synthetic_train.csv and synthetic_test.csv in the data folder.
# 该阶段在数据文件夹中创建了 synthetic_train.csv 和 synthetic_test.csv 文件。
