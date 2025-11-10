# Phase 4 - RAG Pipeline Integration
# This phase integrates the RAG (Retrieval-Augmented Generation) pipeline to enhance explainability for fake review detection. Now using RoBERTa for classification and GPT-2 for explanation generation.
# Depends on synthetic_test.csv from phase 1, model from phase 3, chroma_data from phase 2.

# Initialize results storage
roberta_predictions = []
roberta_confidences = []

# Set batch size for evaluation
batch_size = 64
print(f"Starting RoBERTa evaluation with batch size {batch_size}")
print(f"Total test samples: {len(test_df)}")

# Execute batch evaluation
print("\n[1] RoBERTa Binary Classification - Predictions")

from tqdm import tqdm
import numpy as np

batch_size = 64  # Define batch size

for start_idx in tqdm(range(0, len(test_df), batch_size), desc="Processing RoBERTa batches"):
    end_idx = min(start_idx + batch_size, len(test_df))
    batch_product_infos = test_df.iloc[start_idx:end_idx]['product_info'].tolist()
    batch_review_texts = test_df.iloc[start_idx:end_idx]['review_text'].tolist()

    # Pass tokenizer, model, formatter, device per refactor
    batch_preds, batch_confs = classify_review_roberta_batch(batch_product_infos, batch_review_texts, tokenizer, model, formatter, device)
    roberta_predictions.extend(batch_preds)
    roberta_confidences.extend(batch_confs)

print(f"Completed RoBERTa evaluation: {len(roberta_predictions)} predictions generated")

# Calculate metrics for RoBERTa
ground_truth = test_df['is_fake'].astype(int).tolist()
roberta_metrics = calculate_metrics(np.array(roberta_predictions), np.array(ground_truth), np.array(roberta_confidences))

print("\nRoBERTa Evaluation Metrics:")
for metric, value in roberta_metrics.items():
    if value is not None:
        print(f"{metric.capitalize()}: {value:.4f}")
    else:
        print(f"{metric.capitalize()}: N/A")

# Debug: Check distributions
print(f"\nPredictions distribution: {np.bincount(roberta_predictions)} (0=real, 1=fake)")
print(f"Ground truth distribution: {np.bincount(ground_truth)} (0=real, 1=fake)")
print(f"Sample predictions: {roberta_predictions[:10]}")
print(f"Sample ground truth: {ground_truth[:10]}")

# Define calculate_metrics and print evaluation summary (added to resolve NameError from earlier cell)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def calculate_metrics(predictions, labels, confidences):
    """Calculate evaluation metrics for predictions.

    predictions, labels, confidences are numpy arrays or lists.
    Returns dict with accuracy, precision, recall, f1_score, auc (or None).
    """
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    try:
        auc = roc_auc_score(labels, confidences)
    except Exception:
        auc = None
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc': float(auc) if auc is not None else None,
    }

# Compute and print metrics using variables already present in kernel
try:
    roberta_metrics = calculate_metrics(np.array(roberta_predictions), np.array(ground_truth), np.array(roberta_confidences))
    print("\nRoBERTa Evaluation Metrics:")
    for metric, value in roberta_metrics.items():
        if value is not None:
            print(f"{metric.capitalize()}: {value:.4f}")
        else:
            print(f"{metric.capitalize()}: N/A")
    print(f"\nPredictions distribution: {np.bincount(roberta_predictions)} (0=real, 1=fake)")
    print(f"Ground truth distribution: {np.bincount(ground_truth)} (0=real, 1=fake)")
except Exception as e:
    print('Error computing/printing metrics:', e)

# This phase does not create new files, but evaluates the model and prints results.