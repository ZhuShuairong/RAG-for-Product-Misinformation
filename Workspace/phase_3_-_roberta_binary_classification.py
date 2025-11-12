"""

Phase 3 — RoBERTa binary classification training and evaluation.

This module builds, trains, and evaluates a RoBERTa classifier for fake review detection.

Input: Training and test datasets in CSV format (synthetic_train.csv, synthetic_test.csv), product information (product_info_clean.csv), ChromaDB collection for RAG retrieval.

Processing: Data loading and preprocessing (deduplication, stratified splitting), RAG context precomputation, tokenization, model training with weighted loss for class imbalance, evaluation on test set.

Output: Trained RoBERTa model saved to disk, evaluation metrics (accuracy, precision, recall, F1, AUC), training plots, cached RAG contexts in JSON files.

"""

# Standard library imports
import os
import json
from datetime import datetime
import random
import re
import traceback
from multiprocessing import freeze_support

# Third-party library imports
import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset, load_from_disk, DatasetDict
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import chromadb
import hashlib
import yaml
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt

# Module constants
DEFAULT_SEED = 42
MODEL_NAME = "roberta-base"


class RobertaDatasetFormatter:
    """Format inputs for RoBERTa classification."""

    def __init__(self, tokenizer, max_input_length=256):
        # Initialize tokenizer and max length / 初始化分词器和最大长度
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length

    def format_input(self, product_info, review_text, rag_context=""):
        # Return a human-readable input string for the model / 返回模型的可读输入字符串
        parts = []
        if product_info:
            parts.append(f"Product information: {product_info}")
        if rag_context:
            parts.append(f"Retrieved context: {rag_context}")
        parts.append(f"Review: {review_text}")
        parts.append("\nIs this review authentic for this product?")
        input_text = "\n".join(parts)
        return input_text

    def tokenize_function(self, examples):
        # Tokenize the input texts / 对输入文本进行分词
        inputs = examples["input_text"]
        labels = examples["label"]
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_input_length,
            truncation=True,
            padding="max_length",
        )
        model_inputs["labels"] = labels
        return model_inputs


class LivePlotCallback(TrainerCallback):
    """Custom callback to plot training metrics in real-time."""

    def __init__(self, save_dir='./data/training_plots'):
        # Initialize metrics storage and plotting / 初始化指标存储和绘图
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Initialize metrics storage
        self.train_losses = []
        self.train_steps = []
        self.eval_metrics = {
            'epoch': [],
            'loss': [],
            'accuracy': [],
            'f1': [],
            'auc': []
        }

        # Create figure with subplots
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('RoBERTa Training Progress', fontsize=16, fontweight='bold')

        # Configure subplots
        self.axes[0, 0].set_title('Training Loss')
        self.axes[0, 0].set_xlabel('Steps')
        self.axes[0, 0].set_ylabel('Loss')
        self.axes[0, 0].grid(True, alpha=0.3)

        self.axes[0, 1].set_title('Validation Accuracy')
        self.axes[0, 1].set_xlabel('Epoch')
        self.axes[0, 1].set_ylabel('Accuracy')
        self.axes[0, 1].grid(True, alpha=0.3)
        self.axes[0, 1].set_ylim([0, 1])

        self.axes[1, 0].set_title('Validation F1 Score')
        self.axes[1, 0].set_xlabel('Epoch')
        self.axes[1, 0].set_ylabel('F1 Score')
        self.axes[1, 0].grid(True, alpha=0.3)
        self.axes[1, 0].set_ylim([0, 1])

        self.axes[1, 1].set_title('Validation AUC')
        self.axes[1, 1].set_xlabel('Epoch')
        self.axes[1, 1].set_ylabel('AUC')
        self.axes[1, 1].grid(True, alpha=0.3)
        self.axes[1, 1].set_ylim([0, 1])

        plt.tight_layout()

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Called when the trainer logs metrics / 当训练器记录指标时调用
        if logs is None:
            return

        # Track training loss
        if 'loss' in logs and 'epoch' in logs:
            self.train_losses.append(logs['loss'])
            self.train_steps.append(state.global_step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Called after each evaluation / 每次评估后调用
        if metrics is None:
            return

        # Extract epoch number
        current_epoch = state.epoch

        # Store evaluation metrics
        self.eval_metrics['epoch'].append(current_epoch)
        self.eval_metrics['loss'].append(metrics.get('eval_loss', 0))
        self.eval_metrics['accuracy'].append(metrics.get('eval_accuracy', 0))
        self.eval_metrics['f1'].append(metrics.get('eval_f1', 0))
        self.eval_metrics['auc'].append(metrics.get('eval_auc', 0))

        # Update plots
        self._update_plots()

    def _update_plots(self):
        # Update all subplot visualizations / 更新所有子图可视化
        # Clear previous plots
        for ax in self.axes.flat:
            ax.clear()

        # Plot 1: Training Loss
        if self.train_losses:
            self.axes[0, 0].plot(self.train_steps, self.train_losses, 'b-', linewidth=2, label='Train Loss')
            self.axes[0, 0].set_title('Training Loss')
            self.axes[0, 0].set_xlabel('Steps')
            self.axes[0, 0].set_ylabel('Loss')
            self.axes[0, 0].grid(True, alpha=0.3)
            self.axes[0, 0].legend()

        # Plot 2: Validation Accuracy
        if self.eval_metrics['accuracy']:
            self.axes[0, 1].plot(self.eval_metrics['epoch'], self.eval_metrics['accuracy'],
                                'g-o', linewidth=2, markersize=8, label='Val Accuracy')
            self.axes[0, 1].set_title('Validation Accuracy')
            self.axes[0, 1].set_xlabel('Epoch')
            self.axes[0, 1].set_ylabel('Accuracy')
            self.axes[0, 1].set_ylim([0, 1])
            self.axes[0, 1].grid(True, alpha=0.3)
            self.axes[0, 1].legend()

            # Add value labels
            for i, (x, y) in enumerate(zip(self.eval_metrics['epoch'], self.eval_metrics['accuracy'])):
                self.axes[0, 1].text(x, y + 0.02, f'{y:.3f}', ha='center', fontsize=9)

        # Plot 3: Validation F1
        if self.eval_metrics['f1']:
            self.axes[1, 0].plot(self.eval_metrics['epoch'], self.eval_metrics['f1'],
                                'r-o', linewidth=2, markersize=8, label='Val F1')
            self.axes[1, 0].set_title('Validation F1 Score')
            self.axes[1, 0].set_xlabel('Epoch')
            self.axes[1, 0].set_ylabel('F1 Score')
            self.axes[1, 0].set_ylim([0, 1])
            self.axes[1, 0].grid(True, alpha=0.3)
            self.axes[1, 0].legend()

            # Add value labels
            for i, (x, y) in enumerate(zip(self.eval_metrics['epoch'], self.eval_metrics['f1'])):
                self.axes[1, 0].text(x, y + 0.02, f'{y:.3f}', ha='center', fontsize=9)

        # Plot 4: Validation AUC
        if self.eval_metrics['auc']:
            self.axes[1, 1].plot(self.eval_metrics['epoch'], self.eval_metrics['auc'],
                                'm-o', linewidth=2, markersize=8, label='Val AUC')
            self.axes[1, 1].set_title('Validation AUC')
            self.axes[1, 1].set_xlabel('Epoch')
            self.axes[1, 1].set_ylabel('AUC')
            self.axes[1, 1].set_ylim([0, 1])
            self.axes[1, 1].grid(True, alpha=0.3)
            self.axes[1, 1].legend()

            # Add value labels
            for i, (x, y) in enumerate(zip(self.eval_metrics['epoch'], self.eval_metrics['auc'])):
                self.axes[1, 1].text(x, y + 0.02, f'{y:.3f}', ha='center', fontsize=9)

        plt.tight_layout()

        # Save updated plot
        plot_path = os.path.join(self.save_dir, 'training_progress.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"📊 Updated training plot: {plot_path}")

    def on_train_end(self, args, state, control, **kwargs):
        # Called at the end of training / 训练结束时调用
        # Final plot update
        self._update_plots()

        # Save final metrics to JSON
        metrics_path = os.path.join(self.save_dir, 'training_metrics.json')
        metrics_data = {
            'train_losses': self.train_losses,
            'train_steps': self.train_steps,
            'eval_metrics': self.eval_metrics
        }
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2)
        print(f"📁 Saved training metrics to: {metrics_path}")

        plt.close(self.fig)


def prepare_training_data_roberta(examples, formatter):
    # Prepare lists of input_text and labels from training examples / 从训练示例准备输入文本和标签列表
    inputs = []
    labels = []
    print("Preparing training data for RoBERTa...")
    for i, ex in enumerate(examples):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(examples)} examples")
        input_text = formatter.format_input(
            ex.get('product_info',''),
            ex.get('review_text',''),
            ex.get('rag_context','')
        )
        inputs.append(input_text)
        # FIXED: Handle both 'label' and 'is_fake' column names
        label = ex.get('label', ex.get('is_fake', 0))
        labels.append(int(label))
    return inputs, labels


def compute_metrics(eval_pred):
    # Compute standard classification metrics / 计算标准分类指标
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(labels, torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy())
    except Exception:
        auc = None

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc) if auc is not None else 0.0,
    }


def _safe_hash(text: str) -> str:
    # Create a deterministic short hex id for `text` / 为文本创建确定性的短十六进制ID
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _normalize_text(text: str) -> str:
    # Normalize text for deduplication and overlap checks / 规范化文本用于去重和重叠检查
    if not isinstance(text, str):
        return ''
    txt = text.lower()
    txt = re.sub(r"[^\w\s]", "", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def retrieve_top_k_contexts(collection, query_text, k=3, exclude_product_id=None,
                           max_chars_per_doc=300, max_candidates=6):
    # Query the Chroma collection and return up to k context strings / 查询Chroma集合并返回最多k个上下文字符串
    if collection is None:
        return []

    try:
        results = collection.query(query_texts=[query_text], n_results=max_candidates)
    except Exception as e:
        print(f"RAG query failed: {e}")
        return []

    docs = results.get('documents', [[]])[0] if results.get('documents') else []
    metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else [None] * len(docs)

    selected = []
    for doc, meta in zip(docs, metadatas):
        if doc is None:
            continue
        # Skip exact duplicate of review text
        if query_text.strip() and query_text.strip() in doc:
            continue
        # Skip if metadata product id matches excluded
        if exclude_product_id is not None and meta and isinstance(meta, dict):
            if meta.get('product_id') == exclude_product_id:
                continue

        # Trim
        trimmed = doc.strip()
        if len(trimmed) > max_chars_per_doc:
            trimmed = trimmed[:max_chars_per_doc].rsplit(' ', 1)[0] + '...'
        selected.append(trimmed)
        if len(selected) >= k:
            break

    return selected


def precompute_rag_for_examples(examples, collection, out_path, key_review='review_text',
                                key_product='product_id', k=3, max_chars_per_doc=300,
                                max_candidates=6, force_regenerate=False):
    # Precompute retrieved contexts for examples and save to cache / 预计算示例的检索上下文并保存到缓存
    if examples is None:
        return {}

    # Try loading existing cache
    if os.path.exists(out_path) and not force_regenerate:
        try:
            print(f"Loading existing RAG cache from {out_path}")
            with open(out_path, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            if isinstance(cache, dict):
                for ex in examples:
                    ex_id = ex.get('id', _safe_hash(ex.get(key_review, '')))
                    ex['rag_context'] = cache.get(ex_id, '')
                return cache
        except Exception as e:
            print(f"Warning: failed to load existing RAG cache: {e}")

    print(f"Precomputing RAG contexts for {len(examples)} examples (k={k})...")
    results = {}
    for i, ex in enumerate(examples):
        if i % 200 == 0:
            print(f"  Retrieved {i}/{len(examples)}")
        review = ex.get(key_review, '')
        prod_id = ex.get(key_product)
        ex_id = ex.get('id', _safe_hash(review))
        contexts = retrieve_top_k_contexts(collection, review, k=k,
                                          exclude_product_id=prod_id,
                                          max_chars_per_doc=max_chars_per_doc,
                                          max_candidates=max_candidates)
        joined = ' \n'.join(contexts)
        ex['rag_context'] = joined
        results[ex_id] = joined

    # Save cache
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved RAG cache to {out_path}")
    except Exception as e:
        print(f"Warning: failed to save RAG cache: {e}")

    return results


def main():
    # Main function for training and evaluation / 训练和评估的主函数
    freeze_support()

    # Device setup / 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print("GPU cache cleared")

    # Reduce TF verbosity / 减少TF冗余
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

    # Load training examples / 加载训练示例
    with open('./data/training_examples.json', 'r', encoding='utf-8') as f:
        training_examples = json.load(f)

    # Configuration / 配置
    # Load YAML config / 加载YAML配置
    config_path = os.path.join('Workspace', 'roberta_training_config.yaml')
    if not os.path.exists(config_path):
        config_path = os.path.join('.', 'Workspace', 'roberta_training_config.yaml')

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
        print(f"Loaded training config from {config_path}")
    except Exception as e:
        cfg = {}
        print(f"Warning: could not load YAML config {config_path}: {e}. Using defaults.")

    # Global deterministic seed / 全局确定性种子
    seed_cfg = int(cfg.get('training', {}).get('seed', cfg.get('optimization', {}).get('seed', DEFAULT_SEED)))
    np.random.seed(seed_cfg)
    random.seed(seed_cfg)
    torch.manual_seed(seed_cfg)

    # RAG configuration / RAG配置
    rag_cfg = cfg.get('rag', {}) if cfg else {}
    rag_force_regenerate = bool(rag_cfg.get('force_regenerate', False))
    rag_k = int(rag_cfg.get('k', 3))
    rag_max_chars_per_doc = int(rag_cfg.get('max_chars_per_doc', 300))
    rag_max_candidates = int(rag_cfg.get('max_candidates', 6))

    # Data loading / 数据加载
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)

    # Load training examples / 加载训练示例
    train_df = pd.read_csv('./data/synthetic_train.csv')
    test_df = pd.read_csv('./data/synthetic_test.csv')

    # Create validation split / 创建验证分割
    train_df, val_df = train_test_split(
        train_df, test_size=0.2, random_state=42,
        stratify=train_df['is_fake']
    )

    # Verify no overlap / 验证无重叠
    train_ids = set(train_df['unique_id'])
    val_ids = set(val_df['unique_id'])
    test_ids = set(test_df['unique_id'])
    assert len(train_ids & test_ids) == 0, "Data leakage detected!"

    # Load product info / 加载产品信息
    product_info_clean = pd.read_csv('./data/product_info_clean.csv')

    # Merge product info / 合并产品信息
    desired_product_cols = ['product_id', 'product_name', 'brand', 'primary_category',
                           'price', 'ingredients', 'highlights']
    available_cols = [c for c in desired_product_cols if c in product_info_clean.columns]

    if 'product_id' in available_cols:
        merge_cols = available_cols
        test_df = test_df.merge(product_info_clean[merge_cols], on='product_id', how='left')
        added_cols = [c for c in merge_cols if c != 'product_id']
    else:
        print('Warning: product_info_clean does not contain product_id. Skipping merge.')
        added_cols = []

    # Create product_info column / 创建产品信息列
    def create_product_info(row):
        return (f"Product Name: {row.get('product_name', 'N/A')}\n"
                f"Brand: {row.get('brand', 'N/A')}\n"
                f"Category: {row.get('primary_category', 'N/A')}\n"
                f"Price: {row.get('price', 'N/A')}\n"
                f"Ingredients: {row.get('ingredients', 'N/A')}\n"
                f"Highlights: {row.get('highlights', 'N/A')}")

    test_df['product_info'] = test_df.apply(create_product_info, axis=1)

    # Drop temporary columns / 删除临时列
    if added_cols:
        cols_to_drop = [c for c in added_cols if c in test_df.columns]
        if cols_to_drop:
            test_df = test_df.drop(columns=cols_to_drop)
            print(f"Dropped temporary product columns: {cols_to_drop}")

    # Initialize ChromaDB / 初始化ChromaDB
    try:
        client = chromadb.PersistentClient(path="./data/chroma_data")
        product_profile_collection = client.get_collection(name="product_profiles")
        print("ChromaDB product_profiles collection loaded.")
    except Exception as e:
        product_profile_collection = None
        print(f"Warning: Could not open ChromaDB collection: {e}. RAG retrievals disabled.")

    # Data preprocessing / 数据预处理
    print("\n" + "="*70)
    print("DATA PREPROCESSING - REMOVING DATA LEAKAGE")
    print("="*70)

    # Build training DataFrame / 构建训练数据框架
    df = pd.DataFrame(training_examples)
    if 'review_text' not in df.columns:
        df['review_text'] = df.get('review', '')

    # Ensure IDs exist / 确保ID存在
    if 'id' not in df.columns:
        df['id'] = df['review_text'].fillna('').apply(_safe_hash)
    else:
        df['id'] = df['id'].fillna(df['review_text'].fillna('').apply(_safe_hash))

    df['normalized_text'] = df['review_text'].fillna('').apply(_normalize_text)

    # Normalize test data / 规范化测试数据
    test_df['normalized_text'] = test_df['review_text'].fillna('').apply(_normalize_text)
    if 'id' not in test_df.columns:
        test_df['id'] = test_df['review_text'].fillna('').apply(_safe_hash)
    else:
        test_df['id'] = test_df['id'].fillna(test_df['review_text'].fillna('').apply(_safe_hash))

    # Remove train-test overlap / 删除训练-测试重叠
    test_norms = set(test_df['normalized_text'].tolist())
    test_texts_exact = set(test_df['review_text'].fillna('').tolist())

    before_len = len(df)
    before_counts = df['label'].astype(int).value_counts().to_dict()

    print(f"Before overlap removal: {before_len} samples, distribution: {before_counts}")

    # Remove exact matches / 删除精确匹配
    df_clean = df[~df['review_text'].fillna('').isin(test_texts_exact)].copy()
    exact_removed = before_len - len(df_clean)
    print(f"Removed {exact_removed} exact text matches with test set")

    # Remove normalized matches / 删除规范化匹配
    df_clean = df_clean[~df_clean['normalized_text'].isin(test_norms)].copy()
    normalized_removed = before_len - exact_removed - len(df_clean)
    print(f"Removed {normalized_removed} normalized text matches with test set")

    total_removed = before_len - len(df_clean)
    after_counts = df_clean['label'].astype(int).value_counts().to_dict()

    # Ensure both classes exist / 确保两个类别存在
    if len(after_counts) < 2 or any(after_counts.get(c, 0) == 0 for c in [0, 1]):
        print(f"\n{'!'*70}")
        print("ERROR: Overlap removal eliminated entire class!")
        print(f"Before: {before_counts}")
        print(f"After: {after_counts}")
        print("SOLUTION: Need more training data or different test split strategy")
        print(f"{'!'*70}\n")
        raise ValueError("Insufficient data after overlap removal. Cannot train valid model.")

    print(f"Total train-test overlap removed: {total_removed} ({total_removed/before_len*100:.1f}%)")
    print(f"Class distribution after overlap removal: {after_counts}")

    # Deduplicate / 去重
    df = df_clean.copy()
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['normalized_text']).reset_index(drop=True)
    deduped = before_dedup - len(df)
    print(f"\nDeduplicated training data: removed {deduped} internal duplicates")
    print(f"Final training size: {len(df)} ({df['label'].astype(int).value_counts().to_dict()})")

    # Perform stratified split / 执行分层分割
    print(f"\n{'='*70}")
    print("Creating Stratified Train/Validation Split")
    print(f"{'='*70}")

    # Calculate min samples / 计算最小样本
    min_samples_per_class = df['label'].astype(int).value_counts().min()
    print(f"Minimum samples per class: {min_samples_per_class}")

    if min_samples_per_class < 2:
        print(f"\nWARNING: Only {min_samples_per_class} samples for minority class!")
        print("Cannot create valid stratified split. Need at least 2 samples per class.")
        raise ValueError(f"Insufficient minority class samples: {min_samples_per_class}")

    # Calculate safe validation size / 计算安全验证大小
    max_val_size = 0.2  # 20% default
    safe_val_size = min(max_val_size, (min_samples_per_class - 1) / len(df))
    safe_val_size = max(safe_val_size, 2 / len(df))  # At least 2 samples in validation

    print(f"Using validation size: {safe_val_size:.2%} ({int(len(df) * safe_val_size)} samples)")

    try:
        train_df, val_df = train_test_split(
            df,
            test_size=safe_val_size,
            stratify=df['label'].astype(int),
            random_state=seed_cfg
        )
        print(f"✓ Stratified split successful")
    except Exception as e:
        print(f"✗ Stratified split failed: {e}")
        print("Falling back to random split (not recommended)")
        train_df, val_df = train_test_split(
            df,
            test_size=safe_val_size,
            random_state=seed_cfg
        )

    print(f"Train: {len(train_df)} samples {train_df['label'].astype(int).value_counts().to_dict()}")
    print(f"Val: {len(val_df)} samples {val_df['label'].astype(int).value_counts().to_dict()}")

    # Verify no split overlap / 验证分割无重叠
    train_norms = set(train_df['normalized_text'].tolist())
    val_norms = set(val_df['normalized_text'].tolist())
    train_val_overlap = len(train_norms.intersection(val_norms))
    train_test_overlap_verify = len(train_norms.intersection(test_norms))
    val_test_overlap_verify = len(val_norms.intersection(test_norms))

    print(f"\n{'='*70}")
    print("Split Verification")
    print(f"{'='*70}")
    print(f"Train-Val overlap: {train_val_overlap} ({'✓ PASS' if train_val_overlap == 0 else '✗ FAIL'})")
    print(f"Train-Test overlap: {train_test_overlap_verify} ({'✓ PASS' if train_test_overlap_verify == 0 else '✗ FAIL'})")
    print(f"Val-Test overlap: {val_test_overlap_verify} ({'✓ PASS' if val_test_overlap_verify == 0 else '✗ FAIL'})")

    if any([train_val_overlap > 0, train_test_overlap_verify > 0, val_test_overlap_verify > 0]):
        raise ValueError("Data leakage detected after split! Check deduplication logic.")

    # Create examples list / 创建示例列表
    deduped_examples = df.to_dict(orient='records')

    # RAG precomputation / RAG预计算
    rag_train_cache = './data/rag_retrievals_train.json'
    rag_test_cache = './data/rag_retrievals_test.json'

    if product_profile_collection is not None:
        try:
            _ = precompute_rag_for_examples(
                deduped_examples,
                product_profile_collection,
                rag_train_cache,
                k=rag_k,
                max_chars_per_doc=rag_max_chars_per_doc,
                max_candidates=rag_max_candidates,
                force_regenerate=rag_force_regenerate
            )
        except Exception as e:
            print(f"Warning: precomputing RAG for training failed: {e}")

        # Test RAG
        try:
            test_cache_map = {}
            if os.path.exists(rag_test_cache) and not rag_force_regenerate:
                with open(rag_test_cache, 'r', encoding='utf-8') as f:
                    test_cache_map = json.load(f)
                test_df['rag_context'] = test_df['id'].apply(lambda i: test_cache_map.get(i, ''))
            else:
                test_cache_map = {}
                print(f"Precomputing RAG contexts for test set ({len(test_df)} rows)...")
                for i, row in test_df.iterrows():
                    if i % 200 == 0:
                        print(f"  Retrieved test {i}/{len(test_df)}")
                    review = row.get('review_text', '')
                    prod_id = row.get('product_id') if 'product_id' in row else None
                    contexts = retrieve_top_k_contexts(
                        product_profile_collection,
                        review,
                        k=rag_k,
                        exclude_product_id=prod_id,
                        max_chars_per_doc=rag_max_chars_per_doc,
                        max_candidates=rag_max_candidates
                    )
                    joined = ' \n'.join(contexts)
                    test_cache_map[row['id']] = joined

                with open(rag_test_cache, 'w', encoding='utf-8') as f:
                    json.dump(test_cache_map, f, ensure_ascii=False, indent=2)
                print(f"Saved test RAG cache to {rag_test_cache}")

                test_df['rag_context'] = test_df['id'].apply(lambda i: test_cache_map.get(i, ''))

        except Exception as e:
            print(f"Warning: test RAG precompute failed: {e}")
    else:
        print("Skipping RAG precompute because product_profile_collection is not available.")

    # Debug output / 调试输出
    print("\n" + "="*70)
    print("Sample Training Examples")
    print("="*70)
    for i, ex in enumerate(deduped_examples[:3]):
        print(f"Example {i}: Label={ex.get('label')}, "
              f"Product={str(ex.get('product_info',''))[:80]}..., "
              f"Review={str(ex.get('review_text',''))[:80]}...")

    # Label distribution / 标签分布
    labels = df['label'].astype(int).tolist()
    print(f"\nLabel distribution: {np.bincount(labels)} (0=real, 1=fake)")
    print(f"Label ratio: {np.mean(labels):.3f} fake")

    # Model and tokenizer setup / 模型和分词器设置
    print("\n" + "="*70)
    print("MODEL SETUP")
    print("="*70)

    model_name = MODEL_NAME
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Enable gradient checkpointing / 启用梯度检查点
    model.gradient_checkpointing_enable()
    model = model.to(device)

    print(f"Loaded model: {model_name}")
    print(f"Model device: {next(model.parameters()).device}")

    # Instantiate formatter / 实例化格式化器
    try:
        max_len = cfg.get('tokenizer', {}).get('max_input_length', 256)
    except Exception:
        max_len = 256
    formatter = RobertaDatasetFormatter(tokenizer, max_input_length=int(max_len))

    # Show formatted examples / 显示格式化示例
    print("\nSample formatted inputs:")
    for i, ex in enumerate(deduped_examples[:2]):
        product_info = ex.get('product_info', '')
        review_text = ex.get('review_text', '')
        input_text = formatter.format_input(product_info, review_text)
        print(f"--- Example {i} ---")
        print(input_text[:200] + "...")

    # Tokenization / 分词
    print("\n" + "="*70)
    print("TOKENIZATION")
    print("="*70)

    tokenized_dataset_path = "./data/tokenized_roberta_dataset"
    # Make configurable / 使其可配置
    force_regenerate_tokens = bool(cfg.get('tokenizer', {}).get('force_regenerate', False))
    print(f"Tokenization: {'Regenerating' if force_regenerate_tokens else 'Using cache (if available)'}")

    if os.path.exists(tokenized_dataset_path) and not force_regenerate_tokens:
        print("Loading pre-tokenized dataset...")
        tokenized_datasets = load_from_disk(tokenized_dataset_path)
        train_dataset = tokenized_datasets["train"]
        val_dataset = tokenized_datasets["validation"]
    else:
        # Prepare training/validation inputs
        train_examples = train_df.to_dict(orient='records')
        val_examples = val_df.to_dict(orient='records')

        train_inputs, train_labels = prepare_training_data_roberta(train_examples, formatter)
        val_inputs, val_labels = prepare_training_data_roberta(val_examples, formatter)

        train_dataset = Dataset.from_dict({
            "input_text": train_inputs,
            "label": train_labels
        })
        val_dataset = Dataset.from_dict({
            "input_text": val_inputs,
            "label": val_labels
        })

        print("\nTokenizing datasets...")
        tokenized_train_dataset = train_dataset.map(
            formatter.tokenize_function,
            batched=True,
            num_proc=1
        )
        tokenized_val_dataset = val_dataset.map(
            formatter.tokenize_function,
            batched=True,
            num_proc=1
        )

        # Save tokenized datasets
        tokenized_datasets = DatasetDict({
            "train": tokenized_train_dataset,
            "validation": tokenized_val_dataset
        })
        try:
            tokenized_datasets.save_to_disk(tokenized_dataset_path)
            print(f"Tokenized datasets saved to {tokenized_dataset_path}")
        except Exception as e:
            print(f"Warning: failed to save tokenized datasets: {e}")

        train_dataset = tokenized_train_dataset
        val_dataset = tokenized_val_dataset

    # Persist split mapping / 持久化分割映射
    try:
        splits_path = './data/dataset_splits.json'
        splits_map = {
            'train': train_df['id'].astype(str).tolist(),
            'validation': val_df['id'].astype(str).tolist(),
            'test': test_df['id'].astype(str).tolist()
        }
        with open(splits_path, 'w', encoding='utf-8') as f:
            json.dump(splits_map, f, ensure_ascii=False, indent=2)
        print(f"Saved dataset split mapping to {splits_path}")
    except Exception as e:
        print(f"Warning: failed to save split mapping: {e}")

    # Training setup / 训练设置
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)

    from sklearn.utils.class_weight import compute_class_weight

    # Determine batch sizes
    dl_cfg = cfg.get('dataloader', {}) if cfg else {}
    if device.type == "cpu":
        train_batch_size = int(dl_cfg.get('per_device_train_batch_size', {}).get('cpu', 4))
        eval_batch_size = int(dl_cfg.get('per_device_eval_batch_size', {}).get('cpu', 4))
        grad_accum_steps = int(dl_cfg.get('gradient_accumulation_steps', {}).get('cpu', 4))
    else:
        train_batch_size = int(dl_cfg.get('per_device_train_batch_size', {}).get('gpu', 16))
        eval_batch_size = int(dl_cfg.get('per_device_eval_batch_size', {}).get('gpu', 64))
        grad_accum_steps = int(dl_cfg.get('gradient_accumulation_steps', {}).get('gpu', 2))

    # Optimization hyperparameters
    opt_cfg = cfg.get('optimization', {}) if cfg else {}
    learning_rate = float(opt_cfg.get('learning_rate', opt_cfg.get('lr', 2e-5)))
    weight_decay = float(opt_cfg.get('weight_decay', 0.01))
    fp16 = bool(opt_cfg.get('fp16', True))
    logging_steps_cfg = int(opt_cfg.get('logging_steps', 50))

    # Training control
    train_cfg = cfg.get('training', {}) if cfg else {}
    num_train_epochs_cfg = int(train_cfg.get('num_train_epochs', 5))
    metric_for_best_model_cfg = train_cfg.get('metric_for_best_model', 'f1')
    early_stopping_cfg = bool(train_cfg.get('early_stopping', True))
    early_stopping_patience_cfg = int(train_cfg.get('early_stopping_patience', 2))

    # Compute class weights with MORE AGGRESSIVE weighting for minority class
    try:
        train_labels_list = [ex['label'] for ex in train_dataset]
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.array([0, 1]),
            y=np.array(train_labels_list)
        )
        # CRITICAL FIX: Multiply minority class weight by 2x for extreme imbalance
        class_weights[1] *= 2.0
        class_weights_dict = {int(c): float(w) for c, w in zip([0, 1], class_weights)}
        print(f"Computed class weights: {class_weights_dict}")
    except Exception as e:
        print(f"Warning: could not compute class weights: {e}")
        class_weights = np.array([1.0, 50.0])  # Manual weights
        print(f"Using manual class weights: {class_weights}")

    # Move class weights to device
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Custom Weighted Trainer
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = getattr(outputs, "logits", None)

            if labels is None:
                loss = getattr(outputs, "loss", None)
                if loss is None:
                    loss = torch.tensor(0.0, device=next(model.parameters()).device)
            else:
                loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
                if logits is None:
                    outputs = model(**inputs)
                    logits = outputs.logits
                loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

            return (loss, outputs) if return_outputs else loss

    # Warmup steps
    warmup_steps_cfg = None
    if 'warmup_ratio' in opt_cfg:
        effective_batch = train_batch_size * grad_accum_steps
        steps_per_epoch = max(1, int(np.ceil(len(train_dataset) / effective_batch)))
        total_training_steps = steps_per_epoch * num_train_epochs_cfg
        warmup_steps_cfg = int(total_training_steps * float(opt_cfg.get('warmup_ratio', 0.06)))
    else:
        warmup_steps_cfg = int(opt_cfg.get('warmup_steps', 200))

    training_args = TrainingArguments(
        output_dir=train_cfg.get('output_dir', './data/fake_review_detector_roberta'),
        num_train_epochs=num_train_epochs_cfg,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        eval_strategy=train_cfg.get('eval_strategy', 'epoch'),
        save_strategy=train_cfg.get('save_strategy', 'epoch'),
        logging_steps=logging_steps_cfg,
        warmup_steps=warmup_steps_cfg,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        logging_dir=opt_cfg.get('logging_dir', './logs'),
        seed=seed_cfg,
        load_best_model_at_end=bool(train_cfg.get('load_best_model_at_end', True)),
        metric_for_best_model=metric_for_best_model_cfg,
        greater_is_better=bool(train_cfg.get('greater_is_better', True)),
        fp16=fp16,
        dataloader_num_workers=int(dl_cfg.get('dataloader_num_workers', 4)),
        dataloader_pin_memory=bool(dl_cfg.get('dataloader_pin_memory', True)),
    )

    # Callbacks / 回调
    callbacks = []

    # Add live plotting / 添加实时绘图
    try:
        live_plot_callback = LivePlotCallback(save_dir='./data/training_plots')
        callbacks.append(live_plot_callback)
        print("✓ Live training plots enabled (updates every epoch)")
    except Exception as e:
        print(f"⚠️ Could not enable live plots: {e}")

    # Add early stopping / 添加提前停止
    if early_stopping_cfg:
        try:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience_cfg))
            print(f"✓ Early stopping enabled (patience={early_stopping_patience_cfg})")
        except Exception:
            callbacks.append(EarlyStoppingCallback())

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )

    print(f"Trainer initialized with weighted loss (minority class weight: {class_weights[1]:.2f}x)")

    # Data quality checks / 数据质量检查
    print("\n" + "="*70)
    print("Final Data Quality Report")
    print("="*70)
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Validation dataset: {len(val_dataset)} samples")
    print(f"Test dataframe: {len(test_df)} samples")

    train_labels_final = [ex['label'] for ex in train_dataset]
    val_labels_final = [ex['label'] for ex in val_dataset]
    # Handle both column names / 处理两个列名
    test_labels_final = test_df.get('label', test_df.get('is_fake', pd.Series([0]*len(test_df)))).astype(int).tolist()

    print(f"\nTrain labels: {np.bincount(train_labels_final)} (0=real, 1=fake)")
    print(f"Val labels: {np.bincount(val_labels_final)} (0=real, 1=fake)")
    print(f"Test labels: {np.bincount(test_labels_final)} (0=real, 1=fake)")

    print(f"\nTrain fake ratio: {np.mean(train_labels_final):.3f}")
    print(f"Val fake ratio: {np.mean(val_labels_final):.3f}")
    print(f"Test fake ratio: {np.mean(test_labels_final):.3f}")

    # Training / 训练
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)

    import traceback
    try:
        train_result = trainer.train()
        best_metric = getattr(trainer.state, 'best_metric', None)
        if best_metric is not None:
            print(f"Training completed! Best {metric_for_best_model_cfg}: {best_metric:.4f}")
        else:
            print("Training completed!")
    except Exception as e:
        print("Training failed with exception:")
        traceback.print_exc()
        raise

    # Save model / 保存模型
    trainer.save_model("./data/fake_review_detector_roberta")
    tokenizer.save_pretrained("./data/fake_review_detector_roberta")
    print("\nRoBERTa model saved successfully to './data/fake_review_detector_roberta'")

    # Test set evaluation / 测试集评估
    print("\n" + "="*70)
    print("TEST SET EVALUATION")
    print("="*70)

    # Prepare test examples / 准备测试示例
    test_examples = test_df.to_dict(orient='records')
    test_inputs, test_labels = prepare_training_data_roberta(test_examples, formatter)

    test_dataset = Dataset.from_dict({
        "input_text": test_inputs,
        "label": test_labels
    })

    print("\nTokenizing test dataset...")
    tokenized_test_dataset = test_dataset.map(
        formatter.tokenize_function,
        batched=True,
        num_proc=1
    )

    # Run predictions / 运行预测
    print("Running test set predictions...")
    test_predictions = trainer.predict(tokenized_test_dataset)

    test_preds = np.argmax(test_predictions.predictions, axis=1)
    test_true = test_predictions.label_ids

    # Compute metrics / 计算指标
    print("\n" + "="*70)
    print("TEST SET RESULTS")
    print("="*70)

    test_acc = accuracy_score(test_true, test_preds)
    test_precision = precision_score(test_true, test_preds, zero_division=0)
    test_recall = recall_score(test_true, test_preds, zero_division=0)
    test_f1 = f1_score(test_true, test_preds, zero_division=0)

    try:
        test_probs = torch.softmax(torch.tensor(test_predictions.predictions), dim=1)[:, 1].numpy()
        test_auc = roc_auc_score(test_true, test_probs)
    except:
        test_auc = 0.0

    print(f"\nTest Accuracy:  {test_acc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall:    {test_recall:.4f}")
    print(f"Test F1:        {test_f1:.4f}")
    print(f"Test AUC:       {test_auc:.4f}")

    print("\nConfusion Matrix (rows=true, cols=pred):")
    cm = confusion_matrix(test_true, test_preds)
    print(cm)
    print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}, TP={cm[1,1]}")

    print("\nDetailed Classification Report:")
    print(classification_report(test_true, test_preds, target_names=['Real', 'Fake']))

    # Save test results
    test_results = {
        'test_accuracy': float(test_acc),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_f1': float(test_f1),
        'test_auc': float(test_auc),
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(
            test_true, test_preds,
            target_names=['Real', 'Fake'],
            output_dict=True
        )
    }

    results_path = './data/test_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2)
    print(f"\n✓ Test results saved to {results_path}")

    print("\n" + "="*70)
    print("TRAINING COMPLETE - MODEL SAVED")
    print("="*70)


if __name__ == "__main__":
    freeze_support()
    main()
