# Phase 0 - Setup, Data Loading, EDA, and Cleaning
# This phase handles the initial setup by installing required packages, downloading the dataset from Kaggle, performing exploratory data analysis (EDA), cleaning and imputing missing values in the product and review datasets, and verifying the cleaned data structures.

import subprocess
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# Install required packages
subprocess.check_call(["pip", "install", "chromadb", "sentence-transformers", "torch", "kaggle", "datasets", "python-dotenv"])

# Setup Kaggle API and download dataset
# Check if dataset already exists
expected_files = [
    'product_info.csv',
    'reviews_0-250.csv',
    'reviews_250-500.csv', 
    'reviews_500-750.csv',
    'reviews_750-1250.csv',
    'reviews_1250-end.csv'
]

if os.path.exists('./data/sephora_data') and all(os.path.exists(f'./data/sephora_data/{f}') for f in expected_files):
    print("Dataset already exists, skipping download!")
else:
    # Download dataset
    load_dotenv()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files('nadyinky/sephora-products-and-skincare-reviews', path='./data/sephora_data', unzip=True)
    print("Dataset downloaded successfully!")

# Import libraries and load data
from transformers import EarlyStoppingCallback

# Initial Exploratory Data Analysis (EDA)
def initial_eda(df, name):
    eda_summary = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.values,
        'Null Count': df.isnull().sum().values,
        'Null %': (df.isnull().sum() / len(df) * 100).round(2).values
    })
    print(f"EDA for {name}:")
    print(eda_summary.to_string(index=False))
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

# Ensure data is loaded as DataFrame
try:
    product_info_loaded = isinstance(product_info, pd.DataFrame)
except NameError:
    product_info_loaded = False

if not product_info_loaded:
    print("Loading product_info from CSV...")
    try:
        product_info = pd.read_csv('./data/sephora_data/product_info.csv')
        print("Product info loaded successfully")
    except FileNotFoundError:
        print("Error: product_info.csv not found. Please run the data download cell first.")
        product_info = pd.DataFrame()  # Empty DataFrame to avoid error

try:
    reviews_loaded = isinstance(reviews, pd.DataFrame)
except NameError:
    reviews_loaded = False

if not reviews_loaded:
    print("Loading reviews from CSV...")
    try:
        review_files = ['./data/sephora_data/reviews_0-250.csv', './data/sephora_data/reviews_250-500.csv', './data/sephora_data/reviews_500-750.csv', './data/sephora_data/reviews_750-1250.csv', './data/sephora_data/reviews_1250-end.csv']
        reviews = pd.concat([pd.read_csv(f) for f in review_files], ignore_index=True)
        print("Reviews loaded successfully")
    except FileNotFoundError:
        print("Error: review files not found. Please run the data download cell first.")
        reviews = pd.DataFrame()  # Empty DataFrame to avoid error

initial_eda(product_info, "Product Information")
initial_eda(reviews, "Product Reviews")

if isinstance(product_info, pd.DataFrame) and len(product_info) > 0:
    print(product_info.describe())
else:
    print("Cannot show describe() - product_info not loaded as DataFrame")

# Product Information Data Cleaning
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

print("Product info cleaned, nulls:", product_info_clean.isnull().sum().sum())

# Product Reviews Data Cleaning
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
print("Reviews cleaned, nulls:", reviews_clean.isnull().sum().sum())

# Final Data Verification and Summary
print("Product Info columns:", len(product_info.columns))
print("Reviews columns:", len(reviews.columns))

# Check columns in the datasets
print("Product Info columns:")
print(product_info.columns.tolist())
print(f"\nProduct Info shape: {product_info.shape}")

print("\nReviews columns:")
print(reviews.columns.tolist())
print(f"\nReviews shape: {reviews.shape}")

print("\nProduct Info Clean columns:")
print(product_info_clean.columns.tolist())

print("\nReviews Clean columns:")
print(reviews_clean.columns.tolist())

# Dependencies: This phase creates cleaned dataframes product_info_clean and reviews_clean in memory. No files saved.