# Phase 3 - RoBERTa Binary Classification + Dual-LLM Explanations (Optimized)
# 第 3 阶段 - RoBERTa 二元分类 + 双 LLM 解释（优化版）
# This phase implements a two-stage pipeline for fake review detection with speed optimizations: Stage 1: RoBERTa Binary Classifier (Optimized) - Uses RoBERTa-base optimized for classification tasks. Stage 2: GPT-2 Explanation Generation - Separate LLM for generating explanations post-classification.
# 该阶段实现了一个两阶段的假评论检测管道，并进行了速度优化：第 1 阶段：RoBERTa 二元分类器（优化版）- 使用针对分类任务优化的 RoBERTa-base。第 2 阶段：GPT-2 解释生成 - 用于分类后生成解释的独立 LLM。
# Depends on training_examples.json from phase 2.
# 依赖于第 2 阶段的 training_examples.json。

import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from datasets import Dataset, load_from_disk, DatasetDict
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime
import chromadb
import os

# Device  # 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    torch.cuda.empty_cache()
    print("GPU cache cleared")


class RobertaDatasetFormatter:
    """
    Format inputs for RoBERTa classification.
    RoBERTa 分类的输入格式。

    Methods
    - format_input(product_info, review_text) -> str
    - tokenize_function(examples) -> dict
    """

    def __init__(self, tokenizer, max_input_length=256):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length

    def format_input(self, product_info, review_text):
        """
        Return a human-readable input string for the model.
        返回一个人类可读的模型输入字符串。
        """
        input_text = f"Product information: {product_info}\nReview: {review_text}\n\nIs this review authentic for this product?"
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
    """
    Prepare lists of input_text and labels from training examples.
    根据训练样本准备 input_text 和 labels 列表。

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
        input_text = formatter.format_input(ex['product_info'], ex['review_text'])
        inputs.append(input_text)
        labels.append(int(ex['label']))
    return inputs, labels


def compute_metrics(eval_pred):
    """
    Compute standard classification metrics given (predictions, labels).
    计算给定（预测结果，标签）的标准分类指标。

    Expects eval_pred: (logits, labels)
    预期 eval_pred：（logits，标签）
    Returns a dict of floats suitable for HF Trainer logging.
    返回一个适用于 HF Trainer 日志记录的浮点数字典。
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
    """
    Tokenize inputs, run the model, and return (pred_labels, confidences, logits).
    对输入进行标记化，运行模型，并返回（pred_labels，confidences，logits）。

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
    """
    Classify a single review and return ('FAKE'|'REAL', confidence)
    对单个评论进行分类并返回结果（'虚假'|'真实'，置信度）

    Uses the shared inference helper for consistency and easier testing.
    使用共享推理辅助函数，以保持一致性和便于测试。
    """
    input_text = formatter.format_input(product_info, review_text)
    preds, confs, _ = _run_model_inference(tokenizer, model, [input_text], device)
    pred = int(preds[0])
    confidence = float(confs[0])
    return ("FAKE" if pred == 1 else "REAL"), confidence


def classify_review_roberta_batch(product_infos, review_texts, tokenizer, model, formatter, device):
    """
    Batch classify multiple reviews and return (pred_labels, confidences).
    批量对多个评论进行分类并返回（pred_labels，confidences）。

    Inputs are parallel lists of product_infos and review_texts.
    输入数据是产品信息列表和评论文本列表。
    """
    input_texts = [formatter.format_input(prod, rev) for prod, rev in zip(product_infos, review_texts)]
    preds, confidences, logits = _run_model_inference(tokenizer, model, input_texts, device)
    return preds.tolist(), confidences.tolist()


def get_rag_context(review_text, product_info=""):
    """
    Get RAG context for a review by querying ChromaDB
    通过查询 ChromaDB 获取评论的 RAG 上下文
    """
    try:
        results = product_profile_collection.query(
            query_texts=[review_text],
            n_results=3
        )
        contexts = results['documents'][0] if results.get('documents') else []
        return " ".join(contexts[:2])
    except Exception as e:
        return f"Error retrieving context: {str(e)}"


def generate_explanation_with_gpt2(product_info, review_text, prediction, rag_context):
    prompt = f"Product: {product_info}\nReview: {review_text}\nContext: {rag_context}\n\nExplain why this review might be {'fake' if prediction == 1 else 'real'}:"
    inputs = gpt2_tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = gpt2_model.generate(
            **inputs,
            max_length=inputs['input_ids'].shape[1] + 100,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=gpt2_tokenizer.eos_token_id,
        )
    explanation = gpt2_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return explanation.strip() if explanation.strip() else "Unable to generate explanation."


def generate_explanation(review_text, prediction, confidence):
    prompt = f"Review: {review_text}\nPrediction: {'Real' if prediction == 0 else 'Fake'}\nConfidence: {confidence:.3f}\n\nExplain why this review is predicted as {'real' if prediction == 0 else 'fake'}:"
    inputs = gpt2_tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = gpt2_model.generate(
            **inputs,
            max_length=inputs['input_ids'].shape[1] + 100,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=gpt2_tokenizer.eos_token_id,
        )
    explanation = gpt2_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return explanation.strip()


def explain_prediction(review_text, prediction, confidence):
    try:
        explanation = generate_explanation(review_text, prediction, confidence)
        return explanation if explanation else "Unable to generate explanation."
    except Exception as e:
        return f"Error generating explanation: {str(e)}"


def setup_batch_evaluation(test_df, batch_size=32):
    predictions = []
    ground_truth = []
    explanations = []
    confidences = []
    print(f"Setting up batch evaluation with batch size {batch_size}")
    print(f"Total test samples: {len(test_df)}")
    return predictions, ground_truth, explanations, confidences, batch_size


def get_rag_contexts_for_fake_reviews(fake_reviews):
    batch_rag_contexts = []
    for review_text in fake_reviews:
        rag_context = get_rag_context(review_text, "")
        batch_rag_contexts.append(rag_context)
    return batch_rag_contexts


def generate_explanations_for_fake_reviews(batch_df, fake_indices, batch_rag_contexts):
    fake_explanations = []
    for i, idx in enumerate(fake_indices):
        row = batch_df.iloc[idx]
        explanation = generate_explanation_with_gpt2(
            row['product_info'],
            row['review_text'],
            1,
            batch_rag_contexts[i],
        )
        fake_explanations.append(explanation)
    return fake_explanations


def combine_batch_explanations(batch_df, batch_predictions, fake_indices, fake_explanations):
    batch_explanations = []
    explanation_idx = 0
    for i in range(len(batch_df)):
        if i in fake_indices:
            batch_explanations.append(fake_explanations[explanation_idx])
            explanation_idx += 1
        else:
            batch_explanations.append("This review appears authentic and matches the product information.")
    return batch_explanations


def evaluate_model(test_df, model, tokenizer, chromadb_client, formatter, batch_size=32):
    predictions, ground_truth, explanations, confidences, _ = setup_batch_evaluation(test_df, batch_size)
    print("Starting batch evaluation...")
    for start_idx in tqdm(range(0, len(test_df), batch_size), desc="Processing batches"):
        end_idx = min(start_idx + batch_size, len(test_df))
        batch_df = test_df.iloc[start_idx:end_idx]
        batch_product_infos = batch_df['product_info'].tolist() if 'product_info' in batch_df.columns else [''] * len(batch_df)
        batch_review_texts = batch_df['review_text'].tolist()
        batch_preds, batch_confs = classify_review_roberta_batch(batch_product_infos, batch_review_texts, tokenizer, model, formatter, device)
        fake_indices = [i for i, pred in enumerate(batch_preds) if pred == 1]
        if fake_indices:
            fake_reviews = batch_df.iloc[fake_indices]['review_text'].tolist()
            batch_rag_contexts = get_rag_contexts_for_fake_reviews(fake_reviews)
            fake_explanations = generate_explanations_for_fake_reviews(batch_df, fake_indices, batch_rag_contexts)
            batch_explanations = combine_batch_explanations(batch_df, batch_preds, fake_indices, fake_explanations)
        else:
            batch_explanations = ["This review appears authentic and matches the product information."] * len(batch_df)
        predictions.extend(batch_preds)
        confidences.extend(batch_confs)
        explanations.extend(batch_explanations)
        ground_truth.extend(batch_df['is_fake'].astype(int).tolist())
    return predictions, ground_truth, explanations, confidences


# Data loading and preparation  # 数据加载和准备
with open('./data/training_examples.json', 'r') as f:
    training_examples = json.load(f)

print(f"Loaded {len(training_examples)} training examples")

client = chromadb.PersistentClient(path="./data/chroma_data")
product_profile_collection = client.get_collection(name="product_profiles")

# Debug: Check training data labels  # 调试：检查训练数据标签
print("Sample training examples:")
for i, ex in enumerate(training_examples[:5]):
    print(f"Example {i}: Label={ex['label']}, Product={ex['product_info'][:100]}..., Review={ex['review_text'][:100]}...")

# Check label distribution  # 检查标签分布
labels = [ex['label'] for ex in training_examples]
print(f"\nLabel distribution: {np.bincount(labels)} (0=real, 1=fake)")
print(f"Label ratio: {np.mean(labels):.3f} fake")

# DEBUG: Create a small balanced debug dataset if possible  # 调试：如果可能，创建一个小的平衡调试数据集
print("\n" + "=" * 50)
print("DEBUG: Creating small balanced dataset")
print("=" * 50)

# Sample small balanced dataset  # 抽样小的平衡数据集
debug_size = 1000  # Target size  # 目标大小
real_examples = [ex for ex in training_examples if ex['label'] == 0]
fake_examples = [ex for ex in training_examples if ex['label'] == 1]

# Determine feasible balanced size  # 确定可行的平衡大小
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

    # COMMENTED OUT: Override training_examples for debugging  # 调试时覆盖 training_examples
    # training_examples = debug_training_examples

print("Using FULL training dataset for production training...")

# Debug: Check what the formatted inputs look like  # 调试：检查格式化后的输入是什么样的
print("\nSample formatted inputs:")

# Ensure `tokenizer` is available. If it's missing, try to load a saved tokenizer from the local `./fake_review_detector_roberta` directory (present in the repo), otherwise fall back to the public 'roberta-base' tokenizer.
# 确保 `tokenizer` 可用。如果缺失，尝试从本地 `./fake_review_detector_roberta` 目录加载已保存的分词器（该目录存在于仓库中），否则回退到公共的 'roberta-base' 分词器。
try:
    tokenizer
except NameError:
    try:
        from transformers import RobertaTokenizer
    except Exception as e:
        raise ImportError("transformers not available: " + str(e))
    import os

    model_dir = os.path.join('.', 'fake_review_detector_roberta')
    if os.path.isdir(model_dir):
        tokenizer = RobertaTokenizer.from_pretrained(model_dir)
        print(f"Loaded tokenizer from {model_dir}")
    else:
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        print("Loaded tokenizer 'roberta-base' as fallback")

# Instantiate the formatter (assumes the class RobertaDatasetFormatter is defined earlier in the notebook)  # 实例化格式化器（假设 RobertaDatasetFormatter 类已在笔记本中定义）
try:
    formatter = RobertaDatasetFormatter(tokenizer)
except NameError:
    raise NameError("RobertaDatasetFormatter is not defined. Please run the cell that defines this class before running this debug cell.")

# Show a few formatted examples and tokenization result  # 显示一些格式化的示例和标记化结果
for i, ex in enumerate(training_examples[:3]):
    product_info = ex.get('product_info', '')
    review_text = ex.get('review_text', '')
    input_text = formatter.format_input(product_info, review_text)
    print(f"--- Example {i} ---")
    print(input_text)
    tokens = tokenizer.tokenize(input_text)
    print("Tokens:", tokens)

# Model and tokenizer setup - Using RoBERTa-base for classification  # 模型和分词器设置 - 使用 RoBERTa-base 进行分类
model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Enable gradient checkpointing for memory efficiency  # 启用梯度检查点以提高内存效率
model.gradient_checkpointing_enable()

# Move model to device  # 将模型移动到设备
model = model.to(device)
print(f"Model moved to device: {next(model.parameters()).device}")

formatter = RobertaDatasetFormatter(tokenizer)

print(f"\nLoaded model: {model_name}")
print(f"Model device: {next(model.parameters()).device}")

# Check if tokenized dataset already exists  # 检查标记化数据集是否已存在
force_regenerate = True

tokenized_dataset_path = "./data/tokenized_roberta_dataset"
if os.path.exists(tokenized_dataset_path) and not force_regenerate:
    print("Loading pre-tokenized dataset...")
    tokenized_datasets = load_from_disk(tokenized_dataset_path)
    train_dataset = tokenized_datasets["train"]
    val_dataset = tokenized_datasets["validation"]
else:
    # Prepare training data  # 准备训练数据
    train_inputs, train_labels = prepare_training_data_roberta(training_examples, formatter)

    train_dataset = Dataset.from_dict({
        "input_text": train_inputs,
        "label": train_labels
    })

    print(f"Training dataset size: {len(train_dataset)}")

    # Train/validation split  # 训练/验证集划分
    print("\n" + "=" * 70)
    print("Creating Train/Validation Split")
    print("=" * 70)

    # Determine whether stratification is feasible  # 确定是否可行进行分层抽样
    try:
        label_counts = np.bincount(train_labels)
        use_stratify = label_counts.min() >= 2
    except Exception:
        use_stratify = False

    if use_stratify:
        stratify_labels = train_labels
        print("Using stratified split for tokenization precompute")
    else:
        stratify_labels = None
        print("Not enough examples for stratified split; using random split")

    train_split_indices, val_split_indices = train_test_split(
        range(len(train_dataset)),
        test_size=0.2,
        stratify=stratify_labels,
        random_state=42
    )

    train_split = train_dataset.select(train_split_indices)
    val_split = train_dataset.select(val_split_indices)

    print(f"Training split size: {len(train_split)}")
    print(f"Validation split size: {len(val_split)}")

    # Tokenize datasets (precompute for speed)  # 标记化数据集（预计算以提高速度）
    print("\nTokenizing datasets (this may take a moment)...")
    tokenized_train_dataset = train_split.map(formatter.tokenize_function, batched=True, num_proc=1)
    tokenized_val_dataset = val_split.map(formatter.tokenize_function, batched=True, num_proc=1)

    # Save tokenized datasets  # 保存标记化数据集
    tokenized_datasets = DatasetDict({
        "train": tokenized_train_dataset,
        "validation": tokenized_val_dataset
    })
    tokenized_datasets.save_to_disk(tokenized_dataset_path)
    print(f"Tokenized datasets saved to {tokenized_dataset_path}")

    train_dataset = tokenized_train_dataset
    val_dataset = tokenized_val_dataset

# Training arguments and trainer setup (optimized for speed)  # 训练参数和训练器设置（针对速度进行了优化）
from sklearn.utils.class_weight import compute_class_weight

# Determine batch sizes depending on device  # 根据设备确定批量大小
if device.type == "cpu":
    train_batch_size = 4
    eval_batch_size = 4
    grad_accum_steps = 4
else:
    train_batch_size = 64  # Increased from 16  # 从 16 提高
    eval_batch_size = 64  # Increased from 16  # 从 16 提高
    grad_accum_steps = 1  # Reduced from 2  # 从 2 减少

# Compute class weights from the raw training labels (train_labels is defined earlier when preparing inputs)  # 从原始训练标签计算类权重（train_labels 在准备输入时已定义）
try:
    unique_classes = np.unique(train_labels)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=np.array(train_labels))
    class_weights_dict = {int(c): float(w) for c, w in zip([0, 1], class_weights)}
    print(f"Computed class weights: {class_weights_dict}")
except Exception as e:
    print(f"Warning: could not compute class weights (falling back to 1.0): {e}")
    class_weights = np.array([1.0, 1.0])

# Move class weights to device for loss calculation  # 将类权重移动到设备以进行损失计算
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

# Use weighted loss via a custom Trainer to avoid modifying model code  # 通过自定义 Trainer 使用加权损失以避免修改模型代码
from transformers import Trainer


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Custom compute_loss that accepts extra kwargs from HF Trainer.
        自定义 compute_loss 函数，接受来自 HF Trainer 的额外 kwargs。

        This avoids TypeError when the Trainer passes framework-specific keywords, such as `num_items_in_batch`.
        这样可以避免 Trainer 传递框架特定的关键字（例如 `num_items_in_batch`）时出现 TypeError。
        """
        # Trainer already moves tensors to device; labels are in inputs['labels']  # Trainer 已经将张量移动到设备；标签在 inputs['labels'] 中
        labels = inputs.get("labels")

        # Forward pass
        outputs = model(**inputs)
        # If model returned a precomputed loss (some models do), prefer that when labels are missing  # 前向传递
        logits = getattr(outputs, "logits", None)

        if labels is None:
            # If model provided a loss (e.g., when labels are embedded), use it; otherwise fallback to 0  # 如果模型提供了损失（例如，当标签被嵌入时），则使用它；否则回退到 0
            loss = getattr(outputs, "loss", None)
            if loss is None:
                # Fallback: zero tensor on same device as model  # 回退：与模型相同设备上的零张量
                loss = torch.tensor(0.0, device=next(model.parameters()).device)
        else:
            # Use CrossEntropyLoss with class weights for multi-class/binary classification  # 对于多类/二元分类，使用带类权重的 CrossEntropyLoss
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
            # Ensure logits exist  # 确保 logits 存在
            if logits is None:
                outputs = model(**inputs)
                logits = outputs.logits
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


# TrainingArguments - prefer accuracy or a composite metric for early stopping when F1 may be zero early  # 训练参数 - 当 F1 可能很早为零时，优先考虑准确率或复合指标进行早停
training_args = TrainingArguments(
    output_dir="./data/fake_review_detector_roberta",
    num_train_epochs=3,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    gradient_accumulation_steps=grad_accum_steps,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=1,  # Changed from 50 to 1 to see training loss  # 从 50 改为 1 以查看训练损失
    warmup_steps=200,
    weight_decay=0.01,
    logging_dir="./logs",
    seed=42,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",  # Use accuracy (or 'f1' once F1 becomes meaningful)  # 使用准确率（或一旦 F1 变得有意义后使用 'f1'）
    greater_is_better=True,
    # Speed optimizations  # 速度优化
    fp16=True,  # Mixed precision training  # 混合精度训练
    dataloader_num_workers=4,  # Parallel data loading  # 并行数据加载
    dataloader_pin_memory=True,  # Faster GPU transfer  # 更快的 GPU 传输
)

# Instantiate our WeightedTrainer instead of the default Trainer  # 实例化我们的 WeightedTrainer 而不是默认的 Trainer
# Use a plain EarlyStoppingCallback() without keyword args to avoid mismatches across transformer versions  # 使用不带关键字参数的普通 EarlyStoppingCallback() 以避免不同 transformer 版本之间的不匹配
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback()]
)

print("Trainer initialized with weighted loss and metric_for_best_model=\"accuracy\"")

# Debug: Check dataset sizes  # 调试：检查数据集大小
print("Dataset sizes:")
print(f"Train dataset: {len(train_dataset)} samples")
print(f"Validation dataset: {len(val_dataset)} samples")
print(f"Test dataframe: {len(test_df)} samples")

# Check label distributions  # 检查标签分布
train_labels = [ex['label'] for ex in train_dataset]
val_labels = [ex['label'] for ex in val_dataset]
test_labels = test_df['is_fake'].astype(int).tolist()

print(f"\nTrain labels distribution: {np.bincount(train_labels)} (0=real, 1=fake)")
print(f"Val labels distribution: {np.bincount(val_labels)} (0=real, 1=fake)")
print(f"Test labels distribution: {np.bincount(test_labels)} (0=real, 1=fake)")

# Check data quality: duplicates and label leakage  # 检查数据质量：重复和标签泄漏
print("\nData Quality Checks:")

# Check for duplicates in training data (using original training_examples)  # 检查训练数据中的重复项（使用原始的 training_examples）
train_texts = [ex['review_text'] for ex in training_examples[:len(train_dataset)]]  # Match the split size  # 训练集划分大小
val_texts = [ex['review_text'] for ex in training_examples[len(train_dataset):len(train_dataset) + len(val_dataset)]]  # Val split
test_texts = test_df['review_text'].tolist()

print(f"Train duplicates: {len(train_texts) - len(set(train_texts))}")
print(f"Val duplicates: {len(val_texts) - len(set(val_texts))}")
print(f"Test duplicates: {len(test_texts) - len(set(test_texts))}")

# Check for overlap between splits  # 检查划分之间的重叠
train_set = set(train_texts)
val_set = set(val_texts)
test_set = set(test_texts)

train_val_overlap = len(train_set.intersection(val_set))
train_test_overlap = len(train_set.intersection(test_set))
val_test_overlap = len(val_set.intersection(test_set))

print(f"Train-Val overlap: {train_val_overlap}")
print(f"Train-Test overlap: {train_test_overlap}")
print(f"Val-Test overlap: {val_test_overlap}")

# Check label balance  # 检查标签平衡
print(f"\nLabel balance check:")
print(f"Train fake ratio: {np.mean([ex['label'] for ex in train_dataset]):.3f}")
print(f"Val fake ratio: {np.mean([ex['label'] for ex in val_dataset]):.3f}")
print(f"Test fake ratio: {test_df['is_fake'].mean():.3f}")

# Start training  # 开始训练
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
    # Re-raise to surface the error to the notebook if desired  # 如果需要，将错误重新引发以显示在笔记本中
    raise

# Save model  # 保存模型
trainer.save_model("./data/fake_review_detector_roberta")
tokenizer.save_pretrained("./data/fake_review_detector_roberta")
print("RoBERTa model saved successfully to './data/fake_review_detector_roberta'")

# This phase creates files: fake_review_detector_roberta (model directory) and tokenized_roberta_dataset in the data folder.
# 该阶段创建文件：data 文件夹中的 fake_review_detector_roberta（模型目录）和 tokenized_roberta_dataset。
