# Phase 2 - Feature Engineering & RAG Preparation (Edited for Balanced 750/750 Train + 250/250 Test Sets)

# This phase prepares features for the RAG (Retrieval-Augmented Generation) system. 
# It first generates synthetic_train.csv and synthetic_test.csv with balanced fake/real splits:
# - Train: 750 fake + 750 real = 1500 samples
# - Test: 250 fake + 250 real = 500 samples
# Fakes: Shuffle 1000 from fake_reviews.json, sample subsets (with synthetic additional columns to match Sephora structure).
# Reals: Concat all Sephora CSV chunks, sample subsets randomly (full original columns preserved).
# Then, creates comprehensive product profiles, embeds them using Sentence Transformers, 
# stores them in a ChromaDB vector database for efficient retrieval, generates training examples 
# with explanations for fake review detection, and saves them for model fine-tuning.

# Depends on fake_reviews.json from Phase 1, product_info_clean.csv from prior phases, 
# and real review CSVs in data/sephora_data/ (e.g., reviews_0-250.csv to reviews_1250-end.csv).
# Assumes Sephora CSVs have columns: product_id, author_name, review_title, review_text, review_date, rating, is_recommended, helpful_votes.

import pandas as pd
import json
import glob
import random
import numpy as np
from datetime import datetime, timedelta
from faker import Faker  # For synthetic data generation; install if needed: pip install faker
import chromadb
from sentence_transformers import SentenceTransformer
import os

# Initialize Faker for synthetic fields
fake = Faker()

# Step 1: Prepare synthetic_train.csv and synthetic_test.csv
print("Preparing balanced synthetic train and test CSVs...")

# Load and prepare fake reviews (add synthetic columns to match Sephora structure)
fake_json_path = 'Workspace/fake_reviews.json'
if not os.path.exists(fake_json_path):
    raise FileNotFoundError(f"Fake reviews not found at {fake_json_path}. Run Phase 1 first.")

with open(fake_json_path, 'r', encoding='utf-8') as f:
    fake_data = json.load(f)

# Shuffle fakes
random.seed(42)
random.shuffle(fake_data)

# Prepare full fake DataFrame with synthetic additional columns
df_fake_full = []
synthetic_authors = ['Anonymous User', 'Disappointed Buyer', 'Concerned Consumer']  # Simple list; can expand
synthetic_dates = []
base_date = datetime.now() - timedelta(days=365)  # Past year
for _ in range(1000):
    synthetic_dates.append((base_date + timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d'))

for i, fake_entry in enumerate(fake_data):
    # Base columns from fakes
    row = {
        'product_id': fake_entry['product_id'],
        'review_text': fake_entry['review'],
        'rating': fake_entry['rating'],
        'is_fake': 1  # Label fakes
    }
    
    # Add synthetic columns to match Sephora
    row['author_name'] = random.choice(synthetic_authors)
    row['review_title'] = fake_entry['review'][:50] + '...' if len(fake_entry['review']) > 50 else fake_entry['review']  # Truncate as title
    row['review_date'] = synthetic_dates[i]
    row['is_recommended'] = False
    row['helpful_votes'] = random.randint(0, 5)  # Low helpfulness for fakes
    
    df_fake_full.append(row)

df_fake = pd.DataFrame(df_fake_full)
print(f"Prepared {len(df_fake)} fake reviews with full Sephora-like structure.")

# Sample fakes: 750 for train, 250 for test
train_fakes = df_fake.sample(n=750, random_state=42)
test_fakes = df_fake.drop(train_fakes.index).sample(n=250, random_state=42)
print(f"Sampled 750 train fakes and 250 test fakes.")

# Load real reviews from Sephora CSV files (concat all, preserve full structure)
real_csv_pattern = 'data/sephora_data/reviews_*.csv'
real_files = glob.glob(real_csv_pattern)
if not real_files:
    raise FileNotFoundError(f"No real review files found matching {real_csv_pattern}.")

df_real_list = []
for file_path in real_files:
    df_temp = pd.read_csv(file_path)
    # Standardize review column if needed
    if 'review' in df_temp.columns and 'review_text' not in df_temp.columns:
        df_temp = df_temp.rename(columns={'review': 'review_text'})
    df_temp['is_fake'] = 0  # Label reals
    # Ensure all expected columns exist (fill missing with NaN or defaults if needed)
    required_cols = ['product_id', 'author_name', 'review_title', 'review_text', 'review_date', 'rating', 'is_recommended', 'helpful_votes']
    for col in required_cols:
        if col not in df_temp.columns:
            if col == 'is_recommended':
                df_temp[col] = True  # Default for reals
            elif col == 'helpful_votes':
                df_temp[col] = 0
            else:
                df_temp[col] = np.nan
    df_real_list.append(df_temp)
    print(f"Loaded {len(df_temp)} reviews from {file_path}")

df_real = pd.concat(df_real_list, ignore_index=True)
df_real = df_real[required_cols + ['is_fake']]  # Select standard columns + label
df_real = df_real.dropna(subset=['product_id', 'review_text'])  # Clean essentials
if len(df_real) < 1000:
    raise ValueError(f"Insufficient real reviews: {len(df_real)} < 1000. Check Sephora data.")

# Shuffle and sample reals: 750 for train, 250 for test
df_real = df_real.sample(frac=1, random_state=42).reset_index(drop=True)
train_reals = df_real.iloc[:750]
test_reals = df_real.iloc[750:1000]
print(f"Sampled 750 train reals and 250 test reals.")

# Combine into train/test DataFrames
train_df = pd.concat([train_reals, train_fakes], ignore_index=True)
test_df = pd.concat([test_reals, test_fakes], ignore_index=True)

# Shuffle final sets
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Train set: {len(train_df)} samples ({train_df['is_fake'].value_counts().to_dict()})")
print(f"Test set: {len(test_df)} samples ({test_df['is_fake'].value_counts().to_dict()})")

# Save to CSVs (all columns preserved)
os.makedirs('./data', exist_ok=True)
train_df.to_csv('./data/synthetic_train.csv', index=False)
test_df.to_csv('./data/synthetic_test.csv', index=False)
print(f"Saved full-structure CSVs to ./data/")

# Now proceed with original Phase 2 logic (using the new CSVs; relies on product_id and review_text)
print("Proceeding with product profiles and RAG preparation...")

# Load synthetic datasets (for consistency, though already loaded)
train_df = pd.read_csv('./data/synthetic_train.csv')
test_df = pd.read_csv('./data/synthetic_test.csv')
print(f"Loaded training: {len(train_df)}, testing: {len(test_df)} samples")

# Load cleaned product_info from prior phase
product_info_clean_path = './data/product_info_clean.csv'
if not os.path.exists(product_info_clean_path):
    raise FileNotFoundError(f"Product info not found at {product_info_clean_path}. Prepare from prior phases.")
product_info_clean = pd.read_csv(product_info_clean_path)
print("Loaded cleaned product_info")

# Create product profiles
product_profiles = []
product_ids = []

for _, row in product_info_clean.iterrows():
    profile = f"Product Name: {row.get('product_name', 'N/A')}\n"
    profile += f"Brand: {row.get('brand', 'N/A')}\n"
    profile += f"Primary Category: {row.get('primary_category', 'N/A')}\n"
    if 'secondary_category' in row:
        profile += f"Secondary Category: {row.get('secondary_category', 'N/A')}\n"
    if 'tertiary_category' in row:
        profile += f"Tertiary Category: {row.get('tertiary_category', 'N/A')}\n"
    profile += f"Price: {row.get('price', 'N/A')}\n"
    profile += f"Ingredients: {row.get('ingredients', 'N/A')}\n"
    profile += f"Highlights: {row.get('highlights', 'N/A')}\n"
    
    product_profiles.append(profile.strip())
    product_ids.append(str(row['product_id']))  # Ensure string

print(f"Created {len(product_profiles)} product profiles.")

# Initialize SentenceTransformer for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Loaded embedding model.")

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./data/chroma_data")
collection = client.get_or_create_collection(
    name="product_profiles",
    metadata={"hnsw:space": "cosine"}
)

# Check if collection already has data
if len(collection.get())['ids'] == 0:
    print("Embedding and storing product profiles...")
    # Embed in batches for efficiency
    batch_size = 5000
    embeddings = []
    for i in range(0, len(product_profiles), batch_size):
        batch_profiles = product_profiles[i:i+batch_size]
        batch_embeddings = model.encode(batch_profiles).tolist()
        embeddings.extend(batch_embeddings)
    
    # Add to collection
    collection.add(
        embeddings=embeddings,
        documents=product_profiles,
        metadatas=[{"product_id": pid} for pid in product_ids],
        ids=product_ids
    )
    print(f"Stored {len(product_profiles)} profiles in ChromaDB.")
else:
    print("ChromaDB collection already populated. Skipping embedding.")

# Test retrieval
query_embedding = model.encode(["A hydrating moisturizer for dry skin"])
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)
print("Test retrieval successful:", len(results['documents'][0]) if results['documents'] else "No results")

# Heuristic explanation generation function
def generate_heuristic_explanation(product_info, review_text, label):
    if label == 0:  # Real review
        return "This review appears authentic based on consistent language, specific details matching the product, and realistic user experience."
    
    # Fake review: Generate mismatch-based explanation
    mismatches = []
    if "skincare" in product_info.lower() and any(word in review_text.lower() for word in ["hair", "shampoo", "conditioner"]):
        mismatches.append("Category mismatch: Review mentions hair care, but product is skincare.")
    elif "fragrance" in product_info.lower() and any(word in review_text.lower() for word in ["eat", "food", "taste"]):
        mismatches.append("Inappropriate context: Review discusses taste, unsuitable for fragrance.")
    elif any(ing in review_text.lower() for ing in ["paraben", "phthalate", "cancer"]) and "ingredients" in product_info.lower():
        mismatches.append("Exaggerated health claims: Review falsely alarms about ingredients without evidence.")
    elif any(word in review_text.lower() for word in ["recall", "banned", "lawsuit"]):
        mismatches.append("Unsubstantiated legal claims: Review invents recalls or bans not associated with the product.")
    else:
        mismatches.append("Generic inconsistencies: Review shows unnatural phrasing or overly dramatic complaints.")
    
    return "This review is likely fake due to: " + "; ".join(mismatches[:2]) + "."

# Prepare training examples (skip if exists; uses review_text from train_df)
examples_path = './data/training_examples.json'
if os.path.exists(examples_path):
    print(f"Training examples already exist at {examples_path}. Skipping generation.")
else:
    print("Generating training examples...")
    training_examples = []
    skipped = 0
    
    for _, row in train_df.iterrows():
        try:
            # Get product info by merging on product_id
            prod_row = product_info_clean[product_info_clean['product_id'] == row['product_id']]
            if prod_row.empty:
                print(f"Warning: No product info for {row['product_id']}. Skipping.")
                skipped += 1
                continue
            
            prod_info = prod_row.iloc[0]
            product_info_str = f"Product Name: {prod_info.get('product_name', 'N/A')}\nBrand: {prod_info.get('brand', 'N/A')}\nCategory: {prod_info.get('primary_category', 'N/A')}\nPrice: {prod_info.get('price', 'N/A')}\nIngredients: {prod_info.get('ingredients', 'N/A')}\nHighlights: {prod_info.get('highlights', 'N/A')}"
            
            explanation = generate_heuristic_explanation(product_info_str, row['review_text'], row['is_fake'])
            
            example = {
                "product_info": product_info_str,
                "review_text": str(row['review_text']),
                "label": int(row['is_fake']),
                "explanation_template": explanation
            }
            training_examples.append(example)
            
        except Exception as e:
            print(f"Error processing row: {e}. Skipping.")
            skipped += 1
            continue
    
    # Save examples
    with open(examples_path, 'w', encoding='utf-8') as f:
        json.dump(training_examples, f, indent=4, ensure_ascii=False)
    
    print(f"Generated {len(training_examples)} training examples (skipped {skipped}) and saved to {examples_path}")
    
    # Preview samples
    if training_examples:
        print("Sample real (label 0):")
        real_sample = next(ex for ex in training_examples if ex['label'] == 0)
        print(f"Product Info: {real_sample['product_info'][:100]}...")
        print(f"Review: {real_sample['review_text'][:100]}...")
        print(f"Explanation: {real_sample['explanation_template']}")
        
        print("\nSample fake (label 1):")
        fake_sample = next(ex for ex in training_examples if ex['label'] == 1)
        print(f"Product Info: {fake_sample['product_info'][:100]}...")
        print(f"Review: {fake_sample['review_text'][:100]}...")
        print(f"Explanation: {fake_sample['explanation_template']}")

print("Phase 2 completed: Balanced full-structure CSVs prepared, vector DB ready, training examples generated.")
