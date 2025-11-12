"""
Phase 0 — setup, data loading, EDA and cleaning.

This module handles the initial setup, data loading, exploratory data analysis (EDA),
and data cleaning for the Sephora products and reviews dataset.

Input: Raw CSV files from Kaggle dataset 'nadyinky/sephora-products-and-skincare-reviews'.
Processing: Downloads dataset if not present, loads data into DataFrames, performs EDA,
cleans missing values, and saves cleaned DataFrames.
Output: Cleaned CSV files 'product_info_clean.csv' and 'reviews_clean.csv' in ./data directory.
"""

import os
import shutil
import subprocess
import sys
import threading
import time

from dotenv import load_dotenv
import numpy as np
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from transformers import EarlyStoppingCallback


def initial_eda(df, name):
    # Print a compact EDA summary for df / 打印df的紧凑EDA摘要
    eda_summary = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.values,
        'Null Count': df.isnull().sum().values,
        'Null %': (df.isnull().sum() / len(df) * 100).round(2).values
    })
    print(f"EDA for {name}:")
    print(eda_summary.to_string(index=False))
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")


if __name__ == "__main__":
    # Install required packages / 安装所需的包
    subprocess.check_call(["pip", "install", "chromadb", "sentence-transformers", "torch", "kaggle", "datasets", "python-dotenv"])

    # Setup Kaggle API and download dataset / 设置Kaggle API并下载数据集
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
        # Download dataset / 下载数据集
        load_dotenv()
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files('nadyinky/sephora-products-and-skincare-reviews', path='./data/sephora_data', unzip=True)
        print("Dataset downloaded successfully!")

    # Load data / 加载数据
    print("Loading product_info from CSV...")
    try:
        product_info = pd.read_csv('./data/sephora_data/product_info.csv')
        print("Product info loaded successfully")
    except FileNotFoundError:
        print("Error: product_info.csv not found. Please run the data download cell first.")
        product_info = pd.DataFrame()

    print("Loading reviews from CSV...")
    try:
        review_files = ['./data/sephora_data/reviews_0-250.csv', './data/sephora_data/reviews_250-500.csv', './data/sephora_data/reviews_500-750.csv', './data/sephora_data/reviews_750-1250.csv', './data/sephora_data/reviews_1250-end.csv']
        reviews = pd.concat([pd.read_csv(f) for f in review_files], ignore_index=True)
        print("Reviews loaded successfully")
    except FileNotFoundError:
        print("Error: review files not found. Please run the data download cell first.")
        reviews = pd.DataFrame()

    # Perform EDA / 执行EDA
    initial_eda(product_info, "Product Information")
    initial_eda(reviews, "Product Reviews")

    if isinstance(product_info, pd.DataFrame) and len(product_info) > 0:
        print(product_info.describe())
    else:
        print("Cannot show describe() - product_info not loaded as DataFrame")

    # Clean product info / 清理产品信息
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

    # Clean reviews / 清理评论
    reviews_clean = reviews.copy()
    reviews_clean = reviews_clean.drop('Unnamed: 0', axis=1, errors='ignore')
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

    # Final verification / 最终验证
    print("Product Info columns:", len(product_info.columns))
    print("Reviews columns:", len(reviews.columns))

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

    # Save cleaned data / 保存清理后的数据
    os.makedirs('./data', exist_ok=True)
    product_info_clean.to_csv('./data/product_info_clean.csv', index=False)
    reviews_clean.to_csv('./data/reviews_clean.csv', index=False)
    print("Saved product_info_clean.csv and reviews_clean.csv to ./data")

    # Verify packages / 验证包
    required_pkgs = {
        'faker': 'faker',
        'tqdm': 'tqdm',
        'langchain_community': 'langchain_community'
    }
    for pkg_name, import_name in required_pkgs.items():
        try:
            __import__(import_name)
        except Exception:
            print(f"Warning: package '{pkg_name}' may be missing. Install with: pip install {pkg_name}")

    # Check Ollama / 检查Ollama
    if shutil.which('ollama') is None:
        print("Ollama CLI not found. If you plan to use explanations with Ollama, install it from https://ollama.ai and run 'ollama serve'.")
    else:
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
            if 'gemma3:4b' in result.stdout:
                print('Ollama model gemma3:4b is available.')
            else:
                print("Ollama CLI present but gemma3:4b not found. Pull with: ollama pull gemma3:4b")
        except Exception as e:
            print(f"Ollama check failed: {e}")

    # List threads / 列出线程
    try:
        print("\n--- Active threads (name, daemon) ---")
        for t in threading.enumerate():
            print(f"  {t.name!r}, daemon={t.daemon}")
    except Exception as e:
        print("Could not list threads:", e)

    time.sleep(0.2)
    print("Exiting script explicitly.")
    sys.exit(0)
