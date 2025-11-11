# Phase 3 - RoBERTa Binary Classification (Training-only)
# This phase focuses solely on preparing data, training, and saving a RoBERTa binary classifier
# for fake vs. real product reviews. RAG/Ollama explanation generation has been removed.
# Depends on `training_examples.json`, `synthetic_test.csv`, and `product_info_clean.csv` from earlier phases.

import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from datasets import Dataset, load_from_disk, DatasetDict
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime
import os
import chromadb
import hashlib
import yaml
import random
import re

# Reduce TF oneDNN/INFO verbosity when TF is present
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    torch.cuda.empty_cache()
    print("GPU cache cleared")
 

class RobertaDatasetFormatter:
    """Format inputs for RoBERTa classification.
    Methods
    - format_input(product_info, review_text) -> str
    - tokenize_function(examples) -> dict
    """
    def __init__(self, tokenizer, max_input_length=256):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length

    def format_input(self, product_info, review_text, rag_context=""):
        """Return a human-readable input string for the model.

        rag_context: precomputed retrievals (string) to include. Kept concise to avoid overflow.
        """
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

def prepare_training_data_roberta(examples, formatter):
    """Prepare lists of input_text and labels from training examples.
    Returns
    - inputs: list[str]
    - labels: list[int]
    """
    inputs = []
    labels = []
    print("Preparing training data for RoBERTa...")
    for i, ex in enumerate(examples):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(examples)} examples")
        input_text = formatter.format_input(ex.get('product_info',''), ex.get('review_text',''), ex.get('rag_context',''))
        inputs.append(input_text)
        labels.append(int(ex['label']))
    return inputs, labels

def compute_metrics(eval_pred):
    """Compute standard classification metrics given (predictions, labels).
    Expects eval_pred: (logits, labels)
    Returns a dict of floats suitable for HF Trainer logging.
    """
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
        "auc": float(auc) if auc is not None else None,
    }

def _run_model_inference(tokenizer, model, input_texts, device, max_length=256):
    """Tokenize inputs, run the model, and return (pred_labels, confidences, logits).
    - input_texts: list[str]
    - returns: (preds: np.ndarray, confidences: np.ndarray, logits: np.ndarray)
    """
    inputs = tokenizer(input_texts, return_tensors="pt", max_length=max_length, truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.detach().cpu().numpy()
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
        preds = np.argmax(logits, axis=1)
        confidences = probs[np.arange(len(preds)), preds]
    return preds, confidences, logits

def classify_review(product_info, review_text, tokenizer, model, formatter, device):
    """Classify a single review and return ('FAKE'|'REAL', confidence)
    Uses the shared inference helper for consistency and easier testing.
    """
    input_text = formatter.format_input(product_info, review_text)
    preds, confs, _ = _run_model_inference(tokenizer, model, [input_text], device)
    pred = int(preds[0])
    confidence = float(confs[0])
    return ("FAKE" if pred == 1 else "REAL"), confidence

def classify_review_roberta_batch(product_infos, review_texts, tokenizer, model, formatter, device, rag_contexts=None):
    """Batch classify multiple reviews and return (pred_labels, confidences).
    Inputs are parallel lists of product_infos and review_texts.
    """
    if rag_contexts is None:
        input_texts = [formatter.format_input(prod, rev) for prod, rev in zip(product_infos, review_texts)]
    else:
        input_texts = [formatter.format_input(prod, rev, rag) for prod, rev, rag in zip(product_infos, review_texts, rag_contexts)]
    preds, confidences, logits = _run_model_inference(tokenizer, model, input_texts, device)
    return preds.tolist(), confidences.tolist()

def evaluate_model_classification_only(test_df, model, tokenizer, formatter, batch_size=32):
    """Batch classification-only evaluation. Returns (predictions, ground_truth, confidences)."""
    predictions = []
    ground_truth = []
    confidences = []
    print(f"Starting classification-only evaluation with batch size {batch_size}...")
    for start_idx in tqdm(range(0, len(test_df), batch_size), desc="Processing batches"):
        end_idx = min(start_idx + batch_size, len(test_df))
        batch_df = test_df.iloc[start_idx:end_idx]
        batch_product_infos = batch_df['product_info'].tolist()
        batch_review_texts = batch_df['review_text'].tolist()
        if 'rag_context' in batch_df.columns:
            batch_rag_contexts = batch_df['rag_context'].tolist()
        else:
            batch_rag_contexts = None
        batch_preds, batch_confs = classify_review_roberta_batch(batch_product_infos, batch_review_texts, tokenizer, model, formatter, device, rag_contexts=batch_rag_contexts)
        predictions.extend(batch_preds)
        confidences.extend(batch_confs)
        ground_truth.extend(batch_df['is_fake'].astype(int).tolist())
    return predictions, ground_truth, confidences

def main():
    from multiprocessing import freeze_support
    freeze_support()

    # Data loading and preparation
    with open('./data/training_examples.json', 'r', encoding='utf-8') as f:
        training_examples = json.load(f)
    print(f"Loaded {len(training_examples)} training examples")

    # Load test_df from Phase 2 CSV
    test_df = pd.read_csv('./data/synthetic_test.csv')
    print(f"Loaded {len(test_df)} test samples from synthetic_test.csv")

    # Load product_info_clean to merge for test_df
    product_info_clean = pd.read_csv('./data/product_info_clean.csv')

    # Load YAML config early so we can use seed and RAG options before splitting/dedup
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

    # global deterministic seed
    seed_cfg = int(cfg.get('training', {}).get('seed', cfg.get('optimization', {}).get('seed', 42)))
    np.random.seed(seed_cfg)
    random.seed(seed_cfg)

    # RAG configuration (exposed via YAML under top-level key 'rag')
    rag_cfg = cfg.get('rag', {}) if cfg else {}
    rag_force_regenerate = bool(rag_cfg.get('force_regenerate', False))
    rag_k = int(rag_cfg.get('k', 3))
    rag_max_chars_per_doc = int(rag_cfg.get('max_chars_per_doc', 300))
    rag_max_candidates = int(rag_cfg.get('max_candidates', 6))

    # Merge test_df with product_info to add product_info column (matching Phase 2 format)
    # Safely choose which product info columns are available and merge only those to avoid KeyErrors
    desired_product_cols = ['product_id', 'product_name', 'brand', 'primary_category', 'price', 'ingredients', 'highlights']
    available_cols = [c for c in desired_product_cols if c in product_info_clean.columns]
    if 'product_id' in available_cols:
        merge_cols = available_cols
        test_df = test_df.merge(product_info_clean[merge_cols], on='product_id', how='left')
        added_cols = [c for c in merge_cols if c != 'product_id']
        if not added_cols:
            print('Warning: product_info_clean contains only product_id; no additional product fields were merged.')
    else:
        # product_id missing in product_info_clean - skip merge and warn
        print('Warning: product_info_clean does not contain product_id. Skipping merge of product info into test_df.')
        added_cols = []

    # Create 'product_info' column for test_df
    def create_product_info(row):
        return f"Product Name: {row.get('product_name', 'N/A')}\nBrand: {row.get('brand', 'N/A')}\nCategory: {row.get('primary_category', 'N/A')}\nPrice: {row.get('price', 'N/A')}\nIngredients: {row.get('ingredients', 'N/A')}\nHighlights: {row.get('highlights', 'N/A')}"

    test_df['product_info'] = test_df.apply(create_product_info, axis=1)
    # Remove only the product columns that were actually added during merge
    if added_cols:
        cols_to_drop = [c for c in added_cols if c in test_df.columns]
        if cols_to_drop:
            test_df = test_df.drop(columns=cols_to_drop)
            print(f"Dropped temporary product columns: {cols_to_drop}")
    print("Merged test_df with product info.")

    # Initialize ChromaDB client/collection for RAG retrievals
    try:
        client = chromadb.PersistentClient(path="./data/chroma_data")
        product_profile_collection = client.get_collection(name="product_profiles")
        print("ChromaDB product_profiles collection loaded.")
    except Exception as e:
        product_profile_collection = None
        print(f"Warning: Could not open ChromaDB collection: {e}. RAG retrievals will be disabled.")

    def _safe_hash(text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def _normalize_text(text: str) -> str:
        if not isinstance(text, str):
            return ''
        # lower, remove some punctuation, collapse whitespace
        txt = text.lower()
        txt = re.sub(r"[^\w\s]", "", txt)
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt

    def retrieve_top_k_contexts(collection, query_text, k=3, exclude_product_id=None, max_chars_per_doc=300, max_candidates=6):
        """Query the Chroma collection and return up to k context strings trimmed to max_chars_per_doc.

        Filters:
        - Skip docs that contain the exact query_text (to avoid leakage of identical reviews).
        - If metadata contains product_id and equals exclude_product_id, skip (optional).
        """
        if collection is None:
            return []
        try:
            results = collection.query(query_texts=[query_text], n_results=max_candidates)
        except Exception as e:
            print(f"RAG query failed: {e}")
            return []

        # results['documents'] is usually a list of lists
        docs = []
        metadatas = []
        try:
            docs = results.get('documents', [[]])[0]
        except Exception:
            docs = []
        try:
            metadatas = results.get('metadatas', [[]])[0]
        except Exception:
            metadatas = [None] * len(docs)

        selected = []
        for doc, meta in zip(docs, metadatas):
            if doc is None:
                continue
            # skip exact duplicate of review text
            if query_text.strip() and query_text.strip() in doc:
                continue
            # skip if metadata product id matches excluded
            try:
                if exclude_product_id is not None and meta and isinstance(meta, dict) and meta.get('product_id') == exclude_product_id:
                    continue
            except Exception:
                pass

            # trim
            trimmed = doc.strip()
            if len(trimmed) > max_chars_per_doc:
                trimmed = trimmed[:max_chars_per_doc].rsplit(' ', 1)[0] + '...'
            selected.append(trimmed)
            if len(selected) >= k:
                break
        return selected

    def precompute_rag_for_examples(examples, collection, out_path, key_review='review_text', key_product='product_id', k=3, max_chars_per_doc=300, max_candidates=6, force_regenerate=False):
        """Precompute retrieved contexts for a list of examples (dicts) and save to out_path (json).
        Caches by stable example id. Returns mapping id -> rag_context.
        Examples should include an 'id' field (stable). This function will also assign 'rag_context' in-place.
        """
        # ensure examples is a list of dicts
        if examples is None:
            return {}

        # Try loading existing cache mapped by id
        if os.path.exists(out_path) and not force_regenerate:
            try:
                print(f"Loading existing RAG cache from {out_path}")
                with open(out_path, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                if isinstance(cache, dict):
                    # populate examples
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
            contexts = retrieve_top_k_contexts(collection, review, k=k, exclude_product_id=prod_id, max_chars_per_doc=max_chars_per_doc, max_candidates=max_candidates)
            joined = ' \n'.join(contexts)
            ex['rag_context'] = joined
            results[ex_id] = joined

        # save cache as dict id -> rag_context
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Saved RAG cache to {out_path}")
        except Exception as e:
            print(f"Warning: failed to save RAG cache: {e}")
        return results

    # Precompute RAG contexts for deduplicated examples and test set (cache to disk)
    rag_train_cache = './data/rag_retrievals_train.json'
    rag_test_cache = './data/rag_retrievals_test.json'

    # Build training DataFrame, normalize, deduplicate and ensure stable ids
    df = pd.DataFrame(training_examples)
    if 'review_text' not in df.columns:
        df['review_text'] = df.get('review', '')
    # ensure id exists
    if 'id' not in df.columns:
        df['id'] = df['review_text'].fillna('').apply(_safe_hash)
    else:
        df['id'] = df['id'].fillna(df['review_text'].fillna('').apply(_safe_hash))

    df['normalized_text'] = df['review_text'].fillna('').apply(_normalize_text)

    # Normalize test_df and create stable ids for test rows
    test_df['normalized_text'] = test_df['review_text'].fillna('').apply(_normalize_text)
    test_df['id'] = test_df.get('id', test_df['review_text'].fillna('').apply(_safe_hash))

    # Remove any training examples that overlap with test set by normalized text
    test_norms = set(test_df['normalized_text'].tolist())
    before_len = len(df)
    before_counts = df['label'].astype(int).value_counts().to_dict()

    # try normalized-text based removal first (aggressive)
    df_candidate = df[~df['normalized_text'].isin(test_norms)].copy()
    removed = before_len - len(df_candidate)
    after_counts = df_candidate['label'].astype(int).value_counts().to_dict()

    # If removal would eliminate an entire class, fall back to exact-text removal by review_text
    if any(after_counts.get(c, 0) == 0 for c in [0, 1]):
        print("Warning: normalized-text overlap removal would remove all examples of one class. Falling back to exact-text removal by review_text.")
        test_texts_set = set(test_df['review_text'].fillna('').tolist())
        df_candidate2 = df[~df['review_text'].fillna('').isin(test_texts_set)].copy()
        removed2 = before_len - len(df_candidate2)
        after_counts2 = df_candidate2['label'].astype(int).value_counts().to_dict()
        if any(after_counts2.get(c, 0) == 0 for c in [0, 1]):
            # still problematic: skip removal to preserve class balance and warn
            print("Warning: exact-text removal still removes all examples of one class. Skipping overlap removal to preserve class balance.")
            df = df
            removed_final = 0
            final_counts = before_counts
        else:
            df = df_candidate2
            removed_final = removed2
            final_counts = after_counts2
            print(f"Removed {removed_final} training examples by exact text that overlapped with test set.")
    else:
        df = df_candidate
        removed_final = removed
        final_counts = after_counts
        if removed_final > 0:
            print(f"Removed {removed_final} training examples that overlapped with test set by normalized text.")

    # Deduplicate exact normalized_text duplicates within training data
    before = len(df)
    df = df.drop_duplicates(subset=['normalized_text']).reset_index(drop=True)
    deduped = before - len(df)
    print(f"Deduplicated training examples: removed {deduped} duplicates (normalized_text). Remaining: {len(df)}")

    # Create a deduplicated examples list
    deduped_examples = df.to_dict(orient='records')

    # If ChromaDB available, precompute RAG for deduplicated examples and test set using stable ids
    if product_profile_collection is not None:
        try:
            _ = precompute_rag_for_examples(deduped_examples, product_profile_collection, rag_train_cache, k=rag_k, max_chars_per_doc=rag_max_chars_per_doc, max_candidates=rag_max_candidates, force_regenerate=rag_force_regenerate)
        except Exception as e:
            print(f"Warning: precomputing RAG for training failed: {e}")

        # For test_df, compute rag_context by id
        try:
            # ensure test ids exist
            test_cache_map = {}
            if os.path.exists(rag_test_cache) and not rag_force_regenerate:
                with open(rag_test_cache, 'r', encoding='utf-8') as f:
                    test_cache_map = json.load(f)
                # map into test_df by id
                test_df['rag_context'] = test_df['id'].apply(lambda i: test_cache_map.get(i, ''))
            else:
                # compute and save
                test_cache_map = {}
                print(f"Precomputing RAG contexts for test set ({len(test_df)} rows)...")
                for i, row in test_df.iterrows():
                    if i % 200 == 0:
                        print(f"  Retrieved test {i}/{len(test_df)}")
                    review = row.get('review_text', '')
                    prod_id = row.get('product_id') if 'product_id' in row else None
                    contexts = retrieve_top_k_contexts(product_profile_collection, review, k=rag_k, exclude_product_id=prod_id, max_chars_per_doc=rag_max_chars_per_doc, max_candidates=rag_max_candidates)
                    joined = ' \n'.join(contexts)
                    test_cache_map[row['id']] = joined
                try:
                    with open(rag_test_cache, 'w', encoding='utf-8') as f:
                        json.dump(test_cache_map, f, ensure_ascii=False, indent=2)
                    print(f"Saved test RAG cache to {rag_test_cache}")
                except Exception as e:
                    print(f"Warning: saving test rag cache failed: {e}")
                test_df['rag_context'] = test_df['id'].apply(lambda i: test_cache_map.get(i, ''))
        except Exception as e:
            print(f"Warning: test RAG precompute failed: {e}")
    else:
        print("Skipping RAG precompute because product_profile_collection is not available.")

    # Debug: Check training data labels (after deduplication)
    print("Sample training examples:")
    for i, ex in enumerate(deduped_examples[:5]):
        print(f"Example {i}: Label={ex.get('label')}, Product={str(ex.get('product_info',''))[:100]}..., Review={str(ex.get('review_text',''))[:100]}...")

    # Check label distribution
    try:
        labels = df['label'].astype(int).tolist()
        print(f"\nLabel distribution: {np.bincount(labels)} (0=real, 1=fake)")
        print(f"Label ratio: {np.mean(labels):.3f} fake")
    except Exception:
        print("Warning: could not compute label distribution from deduplicated data.")

    # DEBUG: Create a small balanced debug dataset if possible
    print("\n" + "="*50)
    print("DEBUG: Creating small balanced dataset")
    print("="*50)

    # Sample small balanced dataset
    debug_size = 1000  # Target size
    real_examples = [ex for ex in deduped_examples if int(ex.get('label', 0)) == 0]
    fake_examples = [ex for ex in deduped_examples if int(ex.get('label', 0)) == 1]

    # Determine feasible balanced size
    min_class = min(len(real_examples), len(fake_examples))
    if min_class == 0:
        print("Not enough examples to create a balanced debug dataset; skipping debug sampling.")
    else:
        actual_half = min(min_class, debug_size // 2)
        np.random.seed(42)
        debug_real = np.random.choice(real_examples, size=actual_half, replace=False).tolist()
        debug_fake = np.random.choice(fake_examples, size=actual_half, replace=False).tolist()

        debug_training_examples = debug_real + debug_fake
        np.random.shuffle(debug_training_examples)

        print(f"Debug dataset: {len(debug_training_examples)} examples ({actual_half} real, {actual_half} fake)")

        # COMMENTED OUT: Override training_examples for debugging
        # training_examples = debug_training_examples

    print("Using FULL deduplicated training dataset for production training...")

    # Debug: Check what the formatted inputs look like
    print("\nSample formatted inputs:")

    # Load tokenizer: prefer a local saved tokenizer if present, otherwise use 'roberta-base'
    from transformers import RobertaTokenizer
    model_dir = os.path.join('.', 'fake_review_detector_roberta')
    if os.path.isdir(model_dir):
        tokenizer = RobertaTokenizer.from_pretrained(model_dir)
        print(f"Loaded tokenizer from {model_dir}")
    else:
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        print("Loaded tokenizer 'roberta-base' as fallback")

    # Load YAML config if available
    config_path = os.path.join('Workspace', 'roberta_training_config.yaml')
    if not os.path.exists(config_path):
        config_path = os.path.join('.', 'Workspace', 'roberta_training_config.yaml')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        print(f"Loaded training config from {config_path}")
    except Exception as e:
        cfg = {}
        print(f"Warning: could not load YAML config {config_path}: {e}. Using defaults.")

    # Instantiate the formatter using YAML max_input_length when available
    try:
        max_len = cfg.get('tokenizer', {}).get('max_input_length', 256)
    except Exception:
        max_len = 256
    formatter = RobertaDatasetFormatter(tokenizer, max_input_length=int(max_len))

    # Show a few formatted examples and tokenization result
    for i, ex in enumerate(deduped_examples[:3]):
        product_info = ex.get('product_info', '')
        review_text = ex.get('review_text', '')
        input_text = formatter.format_input(product_info, review_text)
        print(f"--- Example {i} ---")
        print(input_text)
        tokens = tokenizer.tokenize(input_text)
        print("Tokens:", tokens)

    # Model and tokenizer setup - Using RoBERTa-base for classification
    model_name = "roberta-base"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()

    # Move model to device
    model = model.to(device)
    print(f"Model moved to device: {next(model.parameters()).device}")

    formatter = RobertaDatasetFormatter(tokenizer)

    print(f"\nLoaded model: {model_name}")
    print(f"Model device: {next(model.parameters()).device}")

    # Check if tokenized dataset already exists
    force_regenerate = True

    tokenized_dataset_path = "./data/tokenized_roberta_dataset"
    if os.path.exists(tokenized_dataset_path) and not force_regenerate:
        print("Loading pre-tokenized dataset...")
        tokenized_datasets = load_from_disk(tokenized_dataset_path)
        train_dataset = tokenized_datasets["train"]
        val_dataset = tokenized_datasets["validation"]
    else:
        # Split deduplicated DataFrame into train/validation deterministically
        print("\n" + "="*70)
        print("Creating Train/Validation Split on deduplicated data")
        print("="*70)

        try:
            label_counts = df['label'].astype(int).value_counts()
            use_stratify = label_counts.min() >= 2
        except Exception:
            use_stratify = False

        if use_stratify:
            stratify_labels = df['label'].astype(int)
            print("Using stratified split for tokenization precompute")
        else:
            stratify_labels = None
            print("Not enough examples for stratified split; using random split")

        train_df, val_df = train_test_split(
            df,
            test_size=0.2,
            stratify=stratify_labels,
            random_state=seed_cfg
        )

        print(f"Training split size: {len(train_df)}")
        print(f"Validation split size: {len(val_df)}")

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

        # Tokenize datasets (precompute for speed)
        print("\nTokenizing datasets (this may take a moment)...")
        tokenized_train_dataset = train_dataset.map(formatter.tokenize_function, batched=True, num_proc=1)
        tokenized_val_dataset = val_dataset.map(formatter.tokenize_function, batched=True, num_proc=1)

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

        # persist split mapping for reproducibility
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

    # Training arguments and trainer setup (optimized for speed)
    from sklearn.utils.class_weight import compute_class_weight

    # Determine batch sizes and other hyperparameters from YAML (with sensible defaults)
    dl_cfg = cfg.get('dataloader', {}) if cfg else {}
    if device.type == "cpu":
        train_batch_size = int(dl_cfg.get('per_device_train_batch_size', {}).get('cpu', 4))
        eval_batch_size = int(dl_cfg.get('per_device_eval_batch_size', {}).get('cpu', 4))
        grad_accum_steps = int(dl_cfg.get('gradient_accumulation_steps', {}).get('cpu', 4))
    else:
        train_batch_size = int(dl_cfg.get('per_device_train_batch_size', {}).get('gpu', 16))
        eval_batch_size = int(dl_cfg.get('per_device_eval_batch_size', {}).get('gpu', 64))
        grad_accum_steps = int(dl_cfg.get('gradient_accumulation_steps', {}).get('gpu', 2))

    # Other optimization/hyperparameters
    opt_cfg = cfg.get('optimization', {}) if cfg else {}
    learning_rate = float(opt_cfg.get('learning_rate', opt_cfg.get('lr', 2e-5)))
    weight_decay = float(opt_cfg.get('weight_decay', 0.01))
    fp16 = bool(opt_cfg.get('fp16', True))
    logging_steps_cfg = int(opt_cfg.get('logging_steps', 50))
    seed_cfg = int(opt_cfg.get('seed', 42))

    # Training control
    train_cfg = cfg.get('training', {}) if cfg else {}
    num_train_epochs_cfg = int(train_cfg.get('num_train_epochs', 5))
    metric_for_best_model_cfg = train_cfg.get('metric_for_best_model', 'f1')
    early_stopping_cfg = bool(train_cfg.get('early_stopping', True))
    early_stopping_patience_cfg = int(train_cfg.get('early_stopping_patience', train_cfg.get('early_stopping_patience', 2)))

    # Compute class weights from the raw training labels (train_labels is defined earlier when preparing inputs)
    try:
        unique_classes = np.unique(train_labels)
        class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=np.array(train_labels))
        class_weights_dict = {int(c): float(w) for c, w in zip([0, 1], class_weights)}
        print(f"Computed class weights: {class_weights_dict}")
    except Exception as e:
        print(f"Warning: could not compute class weights (falling back to 1.0): {e}")
        class_weights = np.array([1.0, 1.0])

    # Move class weights to device for loss calculation
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Use weighted loss via a custom Trainer to avoid modifying model code
    from transformers import Trainer

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            """Custom compute_loss that accepts extra kwargs from HF Trainer.
            This avoids TypeError when the Trainer passes framework-specific keywords
            such as `num_items_in_batch`.
            """
            # Trainer already moves tensors to device; labels are in inputs['labels']
            labels = inputs.get("labels")

            # Forward pass
            outputs = model(**inputs)
            # If model returned a precomputed loss (some models do), prefer that when labels are missing
            logits = getattr(outputs, "logits", None)

            if labels is None:
                # If model provided a loss (e.g., when labels are embedded), use it; otherwise fallback to 0
                loss = getattr(outputs, "loss", None)
                if loss is None:
                    # Fallback: zero tensor on same device as model
                    loss = torch.tensor(0.0, device=next(model.parameters()).device)
            else:
                # Use CrossEntropyLoss with class weights for multi-class/binary classification
                loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
                # Ensure logits exist
                if logits is None:
                    outputs = model(**inputs)
                    logits = outputs.logits
                loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

            return (loss, outputs) if return_outputs else loss

    # TrainingArguments - prefer accuracy or a composite metric for early stopping when F1 may be zero early
    # Compute warmup_steps if warmup_ratio is specified; otherwise use warmup_steps from config if present
    warmup_steps_cfg = None
    if 'warmup_ratio' in opt_cfg:
        # estimate total steps: ceil(len(train_dataset)/effective_batch) * epochs
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
        logging_dir=opt_cfg.get('logging_dir', './logs'),
        seed=seed_cfg,
        load_best_model_at_end=bool(train_cfg.get('load_best_model_at_end', True)),
        metric_for_best_model=metric_for_best_model_cfg,
        greater_is_better=bool(train_cfg.get('greater_is_better', True)),
        fp16=fp16,
        dataloader_num_workers=int(dl_cfg.get('dataloader_num_workers', 4)),
        dataloader_pin_memory=bool(dl_cfg.get('dataloader_pin_memory', True)),
    )

    # Instantiate our WeightedTrainer instead of the default Trainer
    # Use a plain EarlyStoppingCallback() without keyword args to avoid mismatches across transformer versions
    callbacks = []
    if early_stopping_cfg:
        try:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience_cfg))
        except Exception:
            # Older transformers versions may not accept the kwarg
            callbacks.append(EarlyStoppingCallback())

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )

    print("Trainer initialized with weighted loss and metric_for_best_model=\"accuracy\"")

    # Debug: Check dataset sizes
    print("Dataset sizes:")
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Validation dataset: {len(val_dataset)} samples")
    print(f"Test dataframe: {len(test_df)} samples")

    # Check label distributions
    train_labels = [ex['label'] for ex in train_dataset]
    val_labels = [ex['label'] for ex in val_dataset]
    test_labels = test_df['is_fake'].astype(int).tolist()

    print(f"\nTrain labels distribution: {np.bincount(train_labels)} (0=real, 1=fake)")
    print(f"Val labels distribution: {np.bincount(val_labels)} (0=real, 1=fake)")
    print(f"Test labels distribution: {np.bincount(test_labels)} (0=real, 1=fake)")

    # Check data quality: duplicates and label leakage
    print("\nData Quality Checks:")

    # Recompute deterministic train/validation split on the deduplicated df to report data-quality
    try:
        try:
            stratify_labels = df['label'].astype(int)
            if stratify_labels.value_counts().min() < 2:
                stratify_labels = None
        except Exception:
            stratify_labels = None
        train_df_q, val_df_q = train_test_split(df, test_size=0.2, stratify=stratify_labels, random_state=seed_cfg)
    except Exception:
        # fallback: split by index
        idx = list(range(len(df)))
        split = int(len(df) * 0.8)
        train_df_q = df.iloc[:split]
        val_df_q = df.iloc[split:]

    train_texts = train_df_q['review_text'].fillna('').tolist()
    val_texts = val_df_q['review_text'].fillna('').tolist()
    test_texts = test_df['review_text'].fillna('').tolist()

    print(f"Train duplicates: {len(train_texts) - len(set(train_texts))}")
    print(f"Val duplicates: {len(val_texts) - len(set(val_texts))}")
    print(f"Test duplicates: {len(test_texts) - len(set(test_texts))}")

    # Check for overlap between splits
    train_set = set(train_texts)
    val_set = set(val_texts)
    test_set = set(test_texts)

    train_val_overlap = len(train_set.intersection(val_set))
    train_test_overlap = len(train_set.intersection(test_set))
    val_test_overlap = len(val_set.intersection(test_set))

    print(f"Train-Val overlap: {train_val_overlap}")
    print(f"Train-Test overlap: {train_test_overlap}")
    print(f"Val-Test overlap: {val_test_overlap}")

    # Check label balance
    print(f"\nLabel balance check:")
    print(f"Train fake ratio: {np.mean([ex['label'] for ex in train_dataset]):.3f}")
    print(f"Val fake ratio: {np.mean([ex['label'] for ex in val_dataset]):.3f}")
    print(f"Test fake ratio: {test_df['is_fake'].mean():.3f}")

    # Start training
    import traceback
    print("\nStarting RoBERTa training (optimized)...")
    try:
        train_result = trainer.train()
        best_metric = getattr(trainer.state, 'best_metric', None)
        if best_metric is not None:
            try:
                print(f"Training completed! Best metric: {best_metric:.4f}")
            except Exception:
                print("Training completed! Best metric:", best_metric)
        else:
            print("Training completed! No best_metric available in trainer.state.")
    except Exception as e:
        print("Training failed with exception:")
        traceback.print_exc()
        # Re-raise to surface the error to the notebook if desired
        raise

    # Save model
    trainer.save_model("./data/fake_review_detector_roberta")
    tokenizer.save_pretrained("./data/fake_review_detector_roberta")
    print("RoBERTa model saved successfully to './data/fake_review_detector_roberta'")

    # Optional: Run classification-only evaluation on test set (uncomment to execute)
    # print("\nRunning classification-only evaluation on test set...")
    # predictions, ground_truth, confidences = evaluate_model_classification_only(
    #     test_df, model, tokenizer, formatter, batch_size=32
    # )
    # print(f"Test Accuracy: {accuracy_score(ground_truth, predictions):.4f}")
    # print(f"Test F1: {f1_score(ground_truth, predictions, zero_division=0):.4f}")
    # print("Evaluation complete. Sample predictions saved in variables.")

    # This phase creates files: fake_review_detector_roberta (model directory) and tokenized_roberta_dataset in the data folder.

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
