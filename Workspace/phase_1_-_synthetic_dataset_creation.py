# Phase 1 - Synthetic Dataset Creation for Fake Review Detection
# This phase creates synthetic datasets for training a fake review detection model. It merges cleaned reviews with product information, generates fake reviews by shuffling real review texts, combines real and fake data, and splits into training and testing sets with stratification to ensure balanced labels.
# Depends on cleaned dataframes product_info_clean and reviews_clean from phase 0.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Enhanced Synthetic Data Generation Function (Safer & Faster)
def generate_synthetic_fakes_with_product_swap(merged_df, fake_ratio=0.2, max_rows=None):
    """
    Generate fake reviews by swapping reviews across different product categories while keeping product labels the same.

    This safer version:
    - Optional `max_rows` to operate on a sampled subset for quick runs.
    - Precomputes category groups and all_reviews once.
    - Avoids repeated expensive list constructions.
    - Adds light progress logging and deterministic randomness.
    """
    import numpy as _np
    from tqdm import tqdm

    # Work on a sample if requested to avoid very long runs during interactive debugging
    if max_rows is not None and len(merged_df) > max_rows:
        df = merged_df.sample(n=max_rows, random_state=42).reset_index(drop=True)
        print(f"Using sampled subset of {max_rows} rows for synthetic generation (from {len(merged_df)} total rows)")
    else:
        df = merged_df.reset_index(drop=True)

    _np.random.seed(42)

    # Pre-group by category for efficient sampling
    categories = df['primary_category'].unique()
    category_groups = {cat: df[df['primary_category'] == cat]['review_text'].tolist() for cat in categories}

    # Precompute a flat list of all reviews as fallback
    all_reviews = [rev for lst in category_groups.values() for rev in lst]
    if not all_reviews:
        # If there are no reviews at all, return empty frame
        print("No reviews available to generate fakes from.")
        df['is_fake'] = 0
        return df

    num_fake = max(1, int(fake_ratio * len(df)))
    fake_indices = _np.random.choice(len(df), num_fake, replace=False)

    print(f"Generating {num_fake} fake reviews...")

    # VECTORIZED APPROACH: Precompute categories for all fake indices
    fake_categories = df.loc[fake_indices, 'primary_category'].values
    other_categories_list = []
    available_reviews_list = []

    for cat in tqdm(fake_categories, desc="Processing fake categories"):
        other_cats = [c for c in categories if c != cat]
        if other_cats:
            random_cat = _np.random.choice(other_cats)
            reviews = category_groups.get(random_cat, [])
            # LIMIT REVIEW LIST SIZE FOR EFFICIENCY - sample at most 1000 reviews per category
            if len(reviews) > 1000:
                reviews = _np.random.choice(reviews, size=1000, replace=False).tolist()
            if reviews:
                available_reviews_list.append(reviews)
                other_categories_list.append(random_cat)
            else:
                # Limit fallback list size too
                limited_all_reviews = all_reviews if len(all_reviews) <= 1000 else _np.random.choice(all_reviews, size=1000, replace=False).tolist()
                available_reviews_list.append(limited_all_reviews)
                other_categories_list.append('fallback')
        else:
            # Limit fallback list size too
            limited_all_reviews = all_reviews if len(all_reviews) <= 1000 else _np.random.choice(all_reviews, size=1000, replace=False).tolist()
            available_reviews_list.append(limited_all_reviews)
            other_categories_list.append('fallback')

    # Now generate the fake reviews efficiently - VECTORIZED APPROACH
    print(f"Selecting {len(available_reviews_list)} fake reviews...")

    # VECTORIZED RANDOM SELECTION: Pre-compute all random indices at once for speed
    all_reviews_lengths = [len(reviews) for reviews in available_reviews_list]
    random_indices = _np.random.randint(0, _np.array(all_reviews_lengths))

    new_reviews = []
    for i, reviews in enumerate(tqdm(available_reviews_list, desc="Selecting fake reviews")):
        if len(reviews) > 0:
            new_reviews.append(reviews[random_indices[i]])
        else:
            # Fallback to all_reviews if category is empty
            fallback_idx = _np.random.randint(0, len(all_reviews))
            new_reviews.append(all_reviews[fallback_idx])

    fake_df = df.iloc[fake_indices].copy()
    fake_df['review_text'] = new_reviews
    fake_df['is_fake'] = 1

    real_df = df.drop(index=fake_indices).copy()
    real_df['is_fake'] = 0

    # Return combined dataset (shuffled)
    combined = pd.concat([real_df, fake_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    return combined

# Load and clean data (similar to Phase 0)
print("Loading and cleaning data...")

# Load product_info
try:
    product_info = pd.read_csv('./data/sephora_data/product_info.csv')
    print("Product info loaded successfully")
except FileNotFoundError:
    print("Error: product_info.csv not found.")
    product_info = pd.DataFrame()

# Load reviews
try:
    review_files = ['./data/sephora_data/reviews_0-250.csv', './data/sephora_data/reviews_250-500.csv', './data/sephora_data/reviews_500-750.csv', './data/sephora_data/reviews_750-1250.csv', './data/sephora_data/reviews_1250-end.csv']
    reviews = pd.concat([pd.read_csv(f, low_memory=False) for f in review_files], ignore_index=True)
    print("Reviews loaded successfully")
except FileNotFoundError:
    print("Error: review files not found.")
    reviews = pd.DataFrame()

# Clean product_info
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

# Save cleaned product_info for use in other phases
product_info_clean.to_csv('./data/product_info_clean.csv', index=False)
print("Cleaned product_info saved as 'product_info_clean.csv'")

# Clean reviews
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

# Merge reviews with product information
print(f"Reviews clean shape: {reviews_clean.shape}")
print(f"Product info clean shape: {product_info_clean.shape}")
print(f"Common product_ids: {len(set(reviews_clean['product_id']).intersection(set(product_info_clean['product_id'])))}")

merged_df = pd.merge(reviews_clean, product_info_clean, on='product_id', how='inner', suffixes=('_review', '_product'))
print(f"Merged dataset shape: {merged_df.shape}")
print("Merged columns:", merged_df.columns.tolist()[:10])  # Show first 10

# Create product_info concatenated column
def create_product_info(row):
    # Safely handle both suffixed and unsuffixed column names from the merge
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

# Generate synthetic dataset with category-aware mismatches
# Use max_rows=10000 for faster testing, remove for full dataset
combined_df = generate_synthetic_fakes_with_product_swap(merged_df, fake_ratio=0.5, max_rows=10000)

print(f"Real reviews: {len(combined_df[combined_df['is_fake'] == 0])}")
print(f"Fake reviews: {len(combined_df[combined_df['is_fake'] == 1])}")
print(f"Total combined: {len(combined_df)}")

# Select required columns and rename
name_col = None
if 'product_name_product' in combined_df.columns:
    name_col = 'product_name_product'
elif 'product_name' in combined_df.columns:
    name_col = 'product_name'
else:
    # Fallback: create a product_name column if missing
    combined_df['product_name'] = combined_df.get('product_name_product', 'Unknown')
    name_col = 'product_name'

cols = ['product_id', name_col, 'review_text', 'is_fake', 'product_info']
final_df = combined_df[cols].copy()
final_df.rename(columns={name_col: 'product_name'}, inplace=True)

# Split into training (5000 samples) and testing (5000 samples) with stratification
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

# Save datasets
train_df.to_csv('./data/synthetic_train.csv', index=False)
test_df.to_csv('./data/synthetic_test.csv', index=False)

print("✓ Synthetic datasets saved as 'synthetic_train.csv' and 'synthetic_test.csv'")

# Quick verification
print("\nSample of training data:")
print(train_df.head(2))

# This phase creates files: synthetic_train.csv and synthetic_test.csv in the data folder.