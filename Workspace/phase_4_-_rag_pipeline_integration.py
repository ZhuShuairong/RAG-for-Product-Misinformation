"""Phase 4 — RAG pipeline integration and evaluation.

This module evaluates the trained classifier on the synthetic test set and
integrates retrieval contexts for explainability. It expects artifacts from
phases 1–3 (synthetic_test.csv, a saved RoBERTa model, and ChromaDB
collections).
"""

# Initialize results storage  # 初始化结果存储
roberta_predictions = []
roberta_confidences = []

# Set batch size for evaluation  # 设置评估的批量大小
batch_size = 64
print(f"Starting RoBERTa evaluation with batch size {batch_size}")
print(f"Total test samples: {len(test_df)}")

# Execute batch evaluation  # 执行批量评估
print("\n[1] RoBERTa Binary Classification - Predictions")

from tqdm import tqdm
import numpy as np

batch_size = 64  # Define batch size  # 定义批量大小

for start_idx in tqdm(range(0, len(test_df), batch_size), desc="Processing RoBERTa batches"):
    end_idx = min(start_idx + batch_size, len(test_df))
    batch_product_infos = test_df.iloc[start_idx:end_idx]['product_info'].tolist()
    batch_review_texts = test_df.iloc[start_idx:end_idx]['review_text'].tolist()

    # Pass tokenizer, model, formatter, device per refactor  # 每次传递 tokenizer、model、formatter、device 以进行重构
    batch_preds, batch_confs = classify_review_roberta_batch(batch_product_infos, batch_review_texts, tokenizer, model, formatter, device)
    roberta_predictions.extend(batch_preds)
    roberta_confidences.extend(batch_confs)

print(f"Completed RoBERTa evaluation: {len(roberta_predictions)} predictions generated")

# Calculate metrics for RoBERTa  # 计算 RoBERTa 的指标
ground_truth = test_df['is_fake'].astype(int).tolist()
roberta_metrics = calculate_metrics(np.array(roberta_predictions), np.array(ground_truth), np.array(roberta_confidences))

print("\nRoBERTa Evaluation Metrics:")
for metric, value in roberta_metrics.items():
    if value is not None:
        print(f"{metric.capitalize()}: {value:.4f}")
    else:
        print(f"{metric.capitalize()}: N/A")

# Debug: Check distributions  # 调试：检查分布
print(f"\nPredictions distribution: {np.bincount(roberta_predictions)} (0=real, 1=fake)")
print(f"Ground truth distribution: {np.bincount(ground_truth)} (0=real, 1=fake)")
print(f"Sample predictions: {roberta_predictions[:10]}")
print(f"Sample ground truth: {ground_truth[:10]}")

# Define calculate_metrics and print evaluation summary (added to resolve NameError from earlier cell)  # 定义 calculate_metrics 并打印评估摘要（添加以解决前一单元格中的 NameError）
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def calculate_metrics(predictions, labels, confidences):
    """
    Calculate evaluation metrics for predictions.
    计算预测的评估指标。

    predictions, labels, confidences are numpy arrays or lists.
    预测、标签、置信度是 numpy 数组或列表。
    Returns dict with accuracy, precision, recall, f1_score, auc (or None).
    返回包含准确率、精确率、召回率、F1 分数、AUC（或 None）的字典。
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


# Compute and print metrics using variables already present in kernel  # 使用内核中已存在的变量计算并打印指标
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
# 此阶段不会创建新文件，而是评估模型并打印结果。
