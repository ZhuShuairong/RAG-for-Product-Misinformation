import pandas as pd
import numpy as np

# Load the CSV file
csv_path = 'data/sephora_data/product_info.csv'
try:
    df = pd.read_csv(csv_path)
    print(f"Successfully loaded CSV with shape: {df.shape}")
except FileNotFoundError:
    print(f"ERROR: File not found at {csv_path}. Check the path.")
    exit(1)
except Exception as e:
    print(f"ERROR loading CSV: {e}")
    exit(1)

# 1. Print all column headers
print("\n=== COLUMN HEADERS ===")
print(list(df.columns))

# 2. Data types for all columns
print("\n=== DATA TYPES ===")
print(df.dtypes)

# 3. Basic info summary for the entire DataFrame
print("\n=== GENERAL DATAFRAME INFO ===")
print(df.info())

# 4. Focus on 'product_id': Check for uniqueness, NaNs, and sample values
print("\n=== PRODUCT_ID ANALYSIS ===")
if 'product_id' in df.columns:
    print(f"Unique product_ids count: {df['product_id'].nunique()}")
    print(f"NaN count in product_id: {df['product_id'].isna().sum()}")
    print(f"Sample product_id values (first 20): {df['product_id'].head(20).tolist()}")
    print(f"Data type: {df['product_id'].dtype}")
    # Check if all are numeric-convertible
    def is_numeric_convertible(x):
        if pd.isna(x):
            return False
        try:
            pd.to_numeric(x)
            return True
        except:
            return False
    numeric_mask = df['product_id'].apply(is_numeric_convertible)
    non_numeric_count = (~numeric_mask).sum()
    print(f"Non-numeric convertible values count: {non_numeric_count}")
    if non_numeric_count > 0:
        print("Sample non-numeric values:")
        print(df.loc[~numeric_mask, 'product_id'].head(10).tolist())
else:
    print("WARNING: 'product_id' column not found.")

# 5. Focus on 'reviews': Check for numeric issues, NaNs, and sample values
print("\n=== REVIEWS COLUMN ANALYSIS ===")
if 'reviews' in df.columns:
    print(f"Sample reviews values (first 20): {df['reviews'].head(20).tolist()}")
    print(f"Data type: {df['reviews'].dtype}")
    print(f"NaN count: {df['reviews'].isna().sum()}")
    # Attempt to clean and convert to float (handle common issues like commas)
    cleaned_reviews = df['reviews'].astype(str).str.replace(',', '').str.strip()
    try:
        cleaned_reviews = pd.to_numeric(cleaned_reviews, errors='coerce')
        print(f"After cleaning (commas removed, to numeric): NaN count: {cleaned_reviews.isna().sum()}")
        print(f"Summary stats after cleaning: {cleaned_reviews.describe()}")
        print(f"Zero or negative values count: {(cleaned_reviews <= 0).sum()}")
    except Exception as e:
        print(f"Error in cleaning/conversion: {e}")
else:
    print("WARNING: 'reviews' column not found.")

# 6. Sample rows for overall inspection
print("\n=== SAMPLE DATA (First 5 Rows) ===")
print(df.head())

# 7. Check for other potentially used columns (e.g., product_name, ingredients)
relevant_cols = ['product_name', 'variation_value', 'ingredients', 'highlights', 'primary_category', 'secondary_category', 'tertiary_category']
print("\n=== SAMPLE VALUES FOR PROMPT-RELATED COLUMNS ===")
for col in relevant_cols:
    if col in df.columns:
        print(f"\n{col}:")
        print(f"  NaN count: {df[col].isna().sum()}")
        print(f"  Sample (first 3 non-null): {df[col].dropna().head(3).tolist()}")
    else:
        print(f"{col}: MISSING COLUMN")

# 8. Overall NaN summary
print("\n=== OVERALL MISSING VALUES SUMMARY ===")
print(df.isnull().sum())

print("\n=== END OF DIAGNOSIS ===")
print("Copy and paste the full output above for bug fixing analysis.")
