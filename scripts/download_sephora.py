"""
Script to download Sephora Products and Skincare Reviews dataset from Kaggle
Dataset: https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews
Uses kaggle.json from ./env/ folder
"""

import os
import sys

def setup_kaggle_credentials():
    """Set up Kaggle credentials from ./env/kaggle.json"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_dir = os.path.join(script_dir, 'env')
    kaggle_json_path = os.path.join(env_dir, 'kaggle.json')

    if not os.path.exists(kaggle_json_path):
        print(f"✗ Kaggle credentials not found at: {kaggle_json_path}")
        print("\nPlease:")
        print(f"1. Create an 'env' folder in the same directory as this script")
        print(f"2. Download your kaggle.json from https://www.kaggle.com/me/account")
        print(f"3. Place it at: {kaggle_json_path}")
        return False

    # Set environment variable to use custom config directory
    os.environ['KAGGLE_CONFIG_DIR'] = env_dir
    print(f"✓ Using Kaggle credentials from: {kaggle_json_path}")
    return True

def download_with_kaggle_api():
    """Download using official Kaggle API"""
    try:
        import kaggle

        # Authenticate (will use env/kaggle.json due to KAGGLE_CONFIG_DIR)
        kaggle.api.authenticate()
        print("✓ Authenticated with Kaggle API")

        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)

        # Download and unzip the dataset
        dataset_name = 'nadyinky/sephora-products-and-skincare-reviews'
        print(f"\nDownloading {dataset_name}...")

        kaggle.api.dataset_download_files(
            dataset_name,
            path='data',
            unzip=True
        )

        print("✓ Dataset downloaded successfully to 'data/' folder")

        # List downloaded files
        files = os.listdir('data')
        print(f"\nDownloaded {len(files)} files:")
        for file in sorted(files):
            file_path = os.path.join('data', file)
            size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            print(f"  - {file} ({size:.2f} MB)")

        return True

    except ImportError:
        print("✗ kaggle package not found. Installing...")
        os.system(f"{sys.executable} -m pip install kaggle")
        print("\nPlease run the script again after installation.")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Kaggle Dataset Downloader")
    print("Dataset: Sephora Products and Skincare Reviews")
    print("=" * 60)
    print()

    # Set up credentials from ./env/kaggle.json
    if not setup_kaggle_credentials():
        sys.exit(1)

    print()

    # Download with Kaggle API
    success = download_with_kaggle_api()

    if success:
        print("\n" + "=" * 60)
        print("✓ Download complete!")
        print("=" * 60)
    else:
        print("\n✗ Download failed. Please check your Kaggle credentials.")
        sys.exit(1)