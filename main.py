"""
Main entry point for the project.

Checks whether the `data/` directory contains dataset files. If not, it will
attempt to call functions from `download_sephora.py` to download the dataset.

This file intentionally keeps the logic simple (single if-statement flow).
"""

import os
import sys

try:
    # Import downloader functions from the provided script
    from scripts.download_sephora import setup_kaggle_credentials, download_with_kaggle_api
except Exception:
    # If import fails, we'll still allow the script to run and print a helpful message
    setup_kaggle_credentials = None
    download_with_kaggle_api = None

# Attempt to import Chroma DB helpers (optional)
try:
    from scripts.insert_products import initialize_chroma_db, insert_products
except Exception:
    initialize_chroma_db = None
    insert_products = None

# Attempt to import prepare_context script (optional)
try:
    from scripts.prepare_context import main as prepare_context_main
except Exception:
    prepare_context_main = None


def data_present(data_dir: str) -> bool:
    """Return True if data_dir exists and contains at least one non-empty CSV file."""
    if not os.path.isdir(data_dir):
        return False

    try:
        for name in os.listdir(data_dir):
            if name.lower().endswith('.csv'):
                path = os.path.join(data_dir, name)
                if os.path.isfile(path) and os.path.getsize(path) > 0:
                    return True
    except Exception:
        return False

    return False


def main() -> int:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    # Pipeline steps (idempotent):
    # 1) Ensure data exists (download if missing)
    # 2) Ensure reviews_context.json exists (prepare_context)
    # 3) Ensure Chroma DB is initialized and populated

    # Step 1: Ensure data
    if not data_present(data_dir):
        print(f"Data not found in '{data_dir}'. Will attempt to download using download_sephora.py...")

        # Ensure downloader is available
        if setup_kaggle_credentials is None or download_with_kaggle_api is None:
            print("✗ Could not import downloader (download_sephora.py). Make sure it is present and importable.")
            return 1

        # Set up credentials and download
        if not setup_kaggle_credentials():
            print("✗ Kaggle credentials not set up. Place your kaggle.json in the ./env/ folder and try again.")
            return 2

        success = download_with_kaggle_api()
        if not success:
            print("✗ Download failed. Check messages above for details.")
            return 3

        print("✓ Download completed successfully.")
    else:
        print(f"✓ Data already present in: {data_dir}")

    # Step 2: Ensure reviews context JSON
    rc_code = ensure_reviews_context(script_dir, data_dir)
    if rc_code != 0:
        return rc_code

    # Step 3: Ensure Chroma DB
    return ensure_chroma_db(script_dir, data_dir)


def ensure_reviews_context(script_dir: str, data_dir: str) -> int:
    """Ensure the reviews_context.json file exists; if not, run prepare_context to create it.

    Returns 0 on success, non-zero error code on failure.
    """
    # We look for the reviews context inside the project data directory
    out_path = os.path.join(data_dir, 'reviews_context.json')
    if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
        print(f"✓ Reviews context already present at: {out_path}")
        return 0

    print(f"reviews_context.json not found at {out_path}. Attempting to generate it using scripts/prepare_context.py...")
    if prepare_context_main is None:
        print("✗ prepare_context script not importable (scripts/prepare_context.py). Skipping creation.")
        return 7

    try:
        # Run prepare_context from the repository root so its relative 'data' paths resolve correctly
        cwd_before = os.getcwd()
        try:
            os.chdir(script_dir)
            prepare_context_main()
        finally:
            os.chdir(cwd_before)
    except Exception as e:
        print(f"✗ prepare_context failed: {e}")
        return 8

    if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
        print(f"✓ reviews_context.json generated at: {out_path}")
        return 0

    print("✗ prepare_context did not produce reviews_context.json as expected.")
    return 9


def ensure_chroma_db(script_dir: str, data_dir: str) -> int:
    """Ensure the Chroma DB is initialized and populated if empty.

    Returns an exit code (0 on success, non-zero on failure).
    """
    # --- Chroma DB initialization (only if not already populated) ---
    if initialize_chroma_db is None or insert_products is None:
        print("i) Chroma DB helper functions not available (scripts/insert_products.py). Skipping DB initialization.")
        return 0

    # Initialize or get collection (this will create the persist dir if missing)
    persist_dir = os.path.join(script_dir, 'chroma_db')
    try:
        collection = initialize_chroma_db(persist_directory=persist_dir)
    except Exception as e:
        print(f"✗ Failed to initialize Chroma DB: {e}")
        return 4

    try:
        count = collection.count()
    except Exception:
        count = 0

    if count > 0:
        print(f"✓ Chroma DB already initialized and contains {count} documents. Skipping insert.")
        return 0

    # Insert products from CSV (must exist in data_dir)
    csv_path = os.path.join(data_dir, 'product_info.csv')
    if not os.path.isfile(csv_path):
        print(f"✗ Expected product CSV not found at: {csv_path}. Cannot populate Chroma DB.")
        return 5

    try:
        print("Populating Chroma DB with products from CSV...")
        insert_products(collection, csv_path, batch_size=100)
    except Exception as e:
        print(f"✗ Failed to insert products into Chroma DB: {e}")
        return 6

    print("✓ Chroma DB initialized and populated successfully.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
