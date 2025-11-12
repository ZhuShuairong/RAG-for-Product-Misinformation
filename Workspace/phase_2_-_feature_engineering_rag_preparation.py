# phase_2_-_feature_engineering_rag_preparation_FIXED.py
"""
Fixed version with:
1. Guaranteed no-overlap train/test split
2. Explicit verification checks
3. Safer file handling
"""

import pandas as pd
import json
import glob
import random
import numpy as np
from datetime import datetime, timedelta
from faker import Faker
import chromadb
from sentence_transformers import SentenceTransformer
import os
import hashlib

print("="*70)
print("PHASE 2 - FEATURE ENGINEERING & RAG (FIXED - NO LEAKAGE)")
print("="*70)

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

def safe_hash(text):
    """Create unique ID from text."""
    return hashlib.sha256(str(text).encode('utf-8')).hexdigest()[:16]

# ========== STEP 1: LOAD DATA ==========
print("\nStep 1: Loading data...")

# Load fake reviews from Phase 1
fake_json_path = 'data/fake_reviews.json'
if not os.path.exists(fake_json_path):
    raise FileNotFoundError(f"Fake reviews not found at {fake_json_path}. Run Phase 1 first.")

with open(fake_json_path, 'r', encoding='utf-8') as f:
    fake_data = json.load(f)

print(f"✓ Loaded {len(fake_data)} fake reviews")

# Load real reviews
real_csv_pattern = 'data/sephora_data/reviews*.csv'
real_files = glob.glob(real_csv_pattern)
if not real_files:
    raise FileNotFoundError(f"No real review files found matching {real_csv_pattern}.")

df_real_list = []
for filepath in real_files:
    df_temp = pd.read_csv(filepath)
    if 'review' in df_temp.columns and 'review_text' not in df_temp.columns:
        df_temp = df_temp.rename(columns={'review': 'review_text'})
    df_real_list.append(df_temp)

df_real_all = pd.concat(df_real_list, ignore_index=True)
print(f"✓ Loaded {len(df_real_all)} real reviews from {len(real_files)} files")

# Load product info
product_info_clean_path = './data/product_info_clean.csv'
if not os.path.exists(product_info_clean_path):
    raise FileNotFoundError(f"Product info not found at {product_info_clean_path}.")

product_info_clean = pd.read_csv(product_info_clean_path)
print(f"✓ Loaded product info")

# ========== STEP 2: PREPARE FAKE REVIEWS DATAFRAME ==========
print("\nStep 2: Preparing fake reviews DataFrame...")

faker = Faker()
synthetic_authors = ['Anonymous User', 'Disappointed Buyer', 'Concerned Consumer']
synthetic_dates = []
base_date = datetime.now() - timedelta(days=365)
for _ in range(1000):
    synthetic_dates.append((base_date + timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d'))

df_fake_full = []
for i, fake_entry in enumerate(fake_data):
    row = {
        'product_id': fake_entry['product_id'],
        'review_text': fake_entry['review'],
        'rating': fake_entry['rating'],
        'is_fake': 1,
        'author_name': random.choice(synthetic_authors),
        'review_title': fake_entry['review'][:50] + '...' if len(fake_entry['review']) > 50 else fake_entry['review'],
        'review_date': synthetic_dates[i % len(synthetic_dates)],
        'is_recommended': False,
        'helpful_votes': random.randint(0, 5)
    }
    df_fake_full.append(row)

df_fake = pd.DataFrame(df_fake_full)
print(f"✓ Prepared {len(df_fake)} fake reviews with full structure")

# Add unique IDs to track samples
df_fake['unique_id'] = df_fake['review_text'].apply(safe_hash)

# ========== STEP 3: SPLIT FAKE REVIEWS (CRITICAL FIX) ==========
print("\nStep 3: Splitting fake reviews (NO OVERLAP)...")

desired_train_fakes = 750
desired_test_fakes = 250
total_needed = desired_train_fakes + desired_test_fakes

print(f"Available fake reviews: {len(df_fake)}")
print(f"Needed: {total_needed} (train={desired_train_fakes}, test={desired_test_fakes})")

if len(df_fake) < total_needed:
    print(f"\n{'!'*70}")
    print(f"ERROR: Insufficient fake reviews!")
    print(f"  Available: {len(df_fake)}")
    print(f"  Needed: {total_needed}")
    print(f"  SOLUTION: Run Phase 1 again to generate at least {total_needed} fake reviews")
    print(f"{'!'*70}\n")
    raise ValueError(f"Need {total_needed} fake reviews but only have {len(df_fake)}")

# CRITICAL: Shuffle once, then split sequentially
df_fake_shuffled = df_fake.sample(frac=1, random_state=42).reset_index(drop=True)

train_fakes = df_fake_shuffled.iloc[:desired_train_fakes].copy()
test_fakes = df_fake_shuffled.iloc[desired_train_fakes:total_needed].copy()

print(f"✓ Train fakes: {len(train_fakes)}")
print(f"✓ Test fakes: {len(test_fakes)}")

# VERIFY: No overlap
train_fake_ids = set(train_fakes['unique_id'])
test_fake_ids = set(test_fakes['unique_id'])
overlap_fakes = train_fake_ids.intersection(test_fake_ids)

if len(overlap_fakes) > 0:
    print(f"\n{'!'*70}")
    print(f"ERROR: {len(overlap_fakes)} fake reviews overlap between train and test!")
    print(f"{'!'*70}\n")
    raise ValueError("Data leakage in fake review split!")

print(f"✓ Verified: 0 fake review overlap")

# ========== STEP 4: PREPARE REAL REVIEWS ==========
print("\nStep 4: Preparing real reviews...")

# Standardize real review columns
required_cols = ['product_id', 'author_name', 'review_title', 'review_text', 
                 'review_date', 'rating', 'is_recommended', 'helpful_votes']

for col in required_cols:
    if col not in df_real_all.columns:
        if col == 'is_recommended':
            df_real_all[col] = True
        elif col == 'helpful_votes':
            df_real_all[col] = 0
        else:
            df_real_all[col] = np.nan

df_real = df_real_all[required_cols + ['is_fake'] if 'is_fake' in df_real_all.columns else required_cols].copy()
if 'is_fake' not in df_real.columns:
    df_real['is_fake'] = 0

# Clean
df_real = df_real.dropna(subset=['product_id', 'review_text'])
df_real = df_real[df_real['review_text'].str.len() > 10]  # Filter very short reviews

if len(df_real) < 1000:
    print(f"WARNING: Only {len(df_real)} real reviews available (need 1000)")

# Add unique IDs
df_real['unique_id'] = df_real['review_text'].apply(safe_hash)

# Shuffle and split
df_real_shuffled = df_real.sample(frac=1, random_state=42).reset_index(drop=True)

train_reals = df_real_shuffled.iloc[:750].copy()
test_reals = df_real_shuffled.iloc[750:1000].copy()

print(f"✓ Train reals: {len(train_reals)}")
print(f"✓ Test reals: {len(test_reals)}")

# VERIFY: No overlap
train_real_ids = set(train_reals['unique_id'])
test_real_ids = set(test_reals['unique_id'])
overlap_reals = train_real_ids.intersection(test_real_ids)

if len(overlap_reals) > 0:
    print(f"\n{'!'*70}")
    print(f"ERROR: {len(overlap_reals)} real reviews overlap between train and test!")
    print(f"{'!'*70}\n")
    raise ValueError("Data leakage in real review split!")

print(f"✓ Verified: 0 real review overlap")

# ========== STEP 5: COMBINE AND SHUFFLE ==========
print("\nStep 5: Creating final train/test datasets...")

train_df = pd.concat([train_reals, train_fakes], ignore_index=True)
test_df = pd.concat([test_reals, test_fakes], ignore_index=True)

# Final shuffle
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"✓ Train set: {len(train_df)} samples {dict(train_df['is_fake'].value_counts())}")
print(f"✓ Test set: {len(test_df)} samples {dict(test_df['is_fake'].value_counts())}")

# ========== STEP 6: FINAL VERIFICATION ==========
print(f"\n{'='*70}")
print("FINAL OVERLAP VERIFICATION")
print(f"{'='*70}")

all_train_ids = set(train_df['unique_id'])
all_test_ids = set(test_df['unique_id'])
total_overlap = all_train_ids.intersection(all_test_ids)

print(f"Total train samples: {len(all_train_ids)}")
print(f"Total test samples: {len(all_test_ids)}")
print(f"Overlap: {len(total_overlap)}")

if len(total_overlap) > 0:
    print(f"\n{'!'*70}")
    print(f"CRITICAL ERROR: {len(total_overlap)} samples overlap!")
    print(f"{'!'*70}\n")
    # Print sample overlaps for debugging
    overlap_texts = train_df[train_df['unique_id'].isin(total_overlap)]['review_text'].head(3)
    for i, text in enumerate(overlap_texts, 1):
        print(f"Overlap {i}: {text[:100]}...")
    raise ValueError("Data leakage detected in final datasets!")

print(f"✅ VERIFICATION PASSED: No data leakage detected!")

# ========== STEP 7: SAVE DATASETS ==========
print(f"\n{'='*70}")
print("SAVING DATASETS")
print(f"{'='*70}")

os.makedirs('./data', exist_ok=True)

# Save CSVs
train_df.to_csv('./data/synthetic_train.csv', index=False)
test_df.to_csv('./data/synthetic_test.csv', index=False)
print(f"✓ Saved synthetic_train.csv ({len(train_df)} samples)")
print(f"✓ Saved synthetic_test.csv ({len(test_df)} samples)")

# ========== STEP 8: PREPARE TRAINING EXAMPLES ==========
print(f"\nStep 8: Creating training_examples.json...")

def create_product_info(row):
    return (f"Product Name: {row.get('product_name', 'N/A')}\n"
            f"Brand: {row.get('brand', 'N/A')}\n"
            f"Category: {row.get('primary_category', 'N/A')}\n"
            f"Price: {row.get('price', 'N/A')}\n"
            f"Ingredients: {row.get('ingredients', 'N/A')}\n"
            f"Highlights: {row.get('highlights', 'N/A')}")

def generate_explanation(product_info, review_text, label):
    if label == 0:
        return "This review appears authentic based on consistent language and realistic user experience."
    else:
        return "This review is likely fake due to generic inconsistencies and unnatural phrasing."

training_examples = []
skipped = 0

for _, row in train_df.iterrows():
    try:
        prod_row = product_info_clean[product_info_clean['product_id'] == row['product_id']]
        if prod_row.empty:
            skipped += 1
            continue
        
        prod_info = prod_row.iloc[0]
        product_info_str = create_product_info(prod_info)
        explanation = generate_explanation(product_info_str, row['review_text'], row['is_fake'])
        
        example = {
            'product_id': str(row['product_id']),
            'product_info': product_info_str,
            'review_text': str(row['review_text']),
            'label': int(row['is_fake']),
            'explanation_template': explanation
        }
        training_examples.append(example)
    except Exception as e:
        print(f"Error processing row: {e}")
        skipped += 1

examples_path = './data/training_examples.json'
with open(examples_path, 'w', encoding='utf-8') as f:
    json.dump(training_examples, f, indent=4, ensure_ascii=False)

print(f"✓ Generated {len(training_examples)} training examples (skipped {skipped})")
print(f"✓ Saved to {examples_path}")

# ========== STEP 9: CHROMADB & RAG (optional - keep existing code) ==========
print(f"\nStep 9: ChromaDB and RAG preparation...")
print("(Skipping for brevity - include your existing ChromaDB code here)")

print(f"\n{'='*70}")
print("PHASE 2 COMPLETE - NO DATA LEAKAGE")
print(f"{'='*70}")
print(f"\nGenerated files:")
print(f"  - synthetic_train.csv ({len(train_df)} samples)")
print(f"  - synthetic_test.csv ({len(test_df)} samples)")
print(f"  - training_examples.json ({len(training_examples)} examples)")
print(f"\n✅ All datasets verified leak-free!")
