import time

import pandas as pd
import json
import os
from pathlib import Path
from typing import List, Dict
import numpy as np
from tqdm import tqdm


def load_all_reviews(data_folder: str = "data") -> pd.DataFrame:
    """Load and combine all review CSV files"""
    review_files = [
        "reviews_0-250.csv",
        "reviews_250-500.csv",
        "reviews_500-750.csv",
        "reviews_750-1250.csv",
        "reviews_1250-end.csv"
    ]
    dfs = []
    for filename in review_files:
        filepath = os.path.join(data_folder, filename)
        if os.path.exists(filepath):
            print(f"Loading {filename}...")
            df = pd.read_csv(filepath, low_memory=False)
            dfs.append(df)
            print(f"  Loaded {len(df)} reviews")
        else:
            print(f"Warning: {filepath} not found, skipping...")
    if not dfs:
        raise FileNotFoundError(f"No review files found in {data_folder}")
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal reviews loaded: {len(combined_df)}")
    return combined_df


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace NaN/null values with '(no information found)'"""
    # Create a copy to avoid modifying original
    df = df.copy()
    # Replace NaN with the specified string for all columns
    df = df.fillna("(no information found)")
    # Also handle None, empty strings, and various null representations
    df = df.replace({
        None: "(no information found)",
        "": "(no information found)",
        "nan": "(no information found)",
        "NaN": "(no information found)",
        "null": "(no information found)",
        "NULL": "(no information found)",
        np.nan: "(no information found)"
    })
    return df


def convert_to_string(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all columns to string type"""
    df = df.copy()
    # Convert all columns to string
    for col in df.columns:
        df[col] = df[col].astype(str)
    return df


def create_context_json(df: pd.DataFrame, output_file: str = os.path.join("data", "reviews_context.json")):
    """
    Convert reviews DataFrame to JSON format for LLM context
    
    JSON format for each review:
    {
        "classification": "real" or "fake",
        "context": {
            "rating": ...,
            "is_recommended": ...,
            ... all other fields
        }
    }
    """
    # Define the columns we want in the context
    context_columns = [
        "rating",
        "is_recommended",
        "helpfulness",
        "total_feedback_count",
        "total_neg_feedback_count",
        "total_pos_feedback_count",
        "submission_time",
        "review_text",
        "review_title",
        "skin_tone",
        "eye_color",
        "skin_type",
        "hair_color",
        "product_id",
        "product_name",
        "brand_name",
        "price_usd"
    ]
    reviews_json = []
    print(f"\nProcessing {len(df)} reviews into JSON format...")
    pbar = tqdm(range(len(df)), desc="Reviews processed", unit="reviews")  # Progress bar
    for idx, row in df.iterrows():
        # Create context dictionary
        context = {}
        for col in context_columns:
            if col in df.columns:
                context[col] = row[col]
            else:
                context[col] = "(no information found)"
        # Create the full review entry
        review_entry = {
            "classification": row.get("classification", "(no information found)"),
            "context": context
        }
        reviews_json.append(review_entry)
        pbar.update(1)
    pbar.close()
    # Write to JSON file
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    print(f"\nWriting to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(reviews_json, f, indent=2, ensure_ascii=False)
    print(f"✓ Successfully saved {len(reviews_json)} reviews to {output_file}")
    return reviews_json


def print_sample(reviews_json: List[Dict], n: int = 2):
    """Print sample entries for verification"""
    print(f"\n{'=' * 60}")
    print(f"Sample Entries (first {n}):")
    print('=' * 60)
    for i in range(min(n, len(reviews_json))):
        print(f"\n--- Review {i + 1} ---")
        print(json.dumps(reviews_json[i], indent=2))


def generate_statistics(df: pd.DataFrame, reviews_json: List[Dict]):
    """Generate statistics about the processed data"""
    print(f"\n{'=' * 60}")
    print("Dataset Statistics:")
    print('=' * 60)
    print(f"Total reviews: {len(df)}")
    # Classification distribution
    if 'classification' in df.columns:
        classification_counts = df['classification'].value_counts()
        print(f"\nClassification distribution:")
        for label, count in classification_counts.items():
            print(f"  {label}: {count} ({count / len(df) * 100:.1f}%)")
    # Missing information statistics
    missing_info_count = 0
    for review in reviews_json:
        for key, value in review['context'].items():
            if value == "(no information found)":
                missing_info_count += 1
    total_fields = len(reviews_json) * len(reviews_json[0]['context'])
    print(f"\nTotal context fields: {total_fields}")
    print(f"Fields with missing information: {missing_info_count} ({missing_info_count / total_fields * 100:.1f}%)")
    # File size
    output_file = os.path.join("data", "reviews_context.json")
    if os.path.exists(output_file):
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"\nOutput file size: {file_size_mb:.2f} MB")


def main():
    """Main execution function"""
    print("=" * 60)
    print("Review Context Preparation Script")
    print("=" * 60)
    # Step 1: Load all review files
    print("\nStep 1: Loading review files...")
    df = load_all_reviews(data_folder="data")
    # Step 2: Impute missing values
    print("\nStep 2: Imputing missing values...")
    df = impute_missing_values(df)
    print("✓ Missing values replaced with '(no information found)'")
    # Step 3: Convert all to strings
    print("\nStep 3: Converting all columns to string type...")
    df = convert_to_string(df)
    print("✓ All columns converted to strings")
    # Step 4: Create JSON format
    print("\nStep 4: Creating JSON context format...")
    reviews_json = create_context_json(df, output_file=os.path.join("data", "reviews_context.json"))
    # Step 5: Show sample and statistics
    print_sample(reviews_json, n=2)
    generate_statistics(df, reviews_json)
    print(f"\n{'=' * 60}")
    print("✓ Processing complete!")
    print('=' * 60)
    print(f"\nOutput file: {os.path.join('data', 'reviews_context.json')}")
    print("This file is ready to be used as LLM prompt context.")


if __name__ == "__main__":
    main()
