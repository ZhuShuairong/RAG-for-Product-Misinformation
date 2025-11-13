#!/usr/bin/env python3
"""
download_sephora.py
简单封装 kaggle dataset 下载（需预先配置 kaggle.json）
Usage:
    python scripts/download_sephora.py --dataset nadyinky/sephora-products-and-skincare-reviews --out_dir data
"""
import argparse
import os
import shutil
import sys
from kaggle import api
import zipfile
import time

def download_dataset(dataset: str, out_dir: str = "data", force: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    # check already present
    expected_marker = os.path.join(out_dir, ".downloaded")
    if os.path.exists(expected_marker) and not force:
        print(f"[INFO] dataset already downloaded in {out_dir}. Use --force to redownload.")
        return True
    try:
        print(f"[INFO] Downloading {dataset} into {out_dir} ...")
        api.dataset_download_files(dataset, path=out_dir, unzip=True, quiet=False)
        # create marker
        with open(expected_marker, "w") as f:
            f.write(f"downloaded: {time.ctime()}\n")
        print("[INFO] Download finished.")
        return True
    except Exception as e:
        print("[ERROR] Download failed:", e)
        return False

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="nadyinky/sephora-products-and-skincare-reviews")
    p.add_argument("--out_dir", type=str, default="data")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    ok = download_dataset(args.dataset, args.out_dir, args.force)
    if not ok:
        sys.exit(1)
