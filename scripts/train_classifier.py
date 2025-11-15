import argparse
import json
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer,
                          RobertaForSequenceClassification, get_scheduler, EarlyStoppingCallback)
import numpy as np
from tqdm import tqdm
import sklearn
import torch
import random


def load_jsonl(file_path, max_rows=None):
    """Load the JSONL file with the reviews and labels."""
    rows = []
    with open(file_path, "r", encoding="utf-8") as fin:
        total_lines = sum(1 for _ in fin)
    with open(file_path, "r", encoding="utf-8") as fin:
        for i, line in enumerate(tqdm(fin, total=total_lines, desc="Loading data")):
            if max_rows and i >= max_rows:
                break
            rows.append(json.loads(line))
    return rows


def prepare_examples(rows, max_retrieved_chars=None):
    """Prepare examples for the model, including context and product information."""
    examples = []
    real_count, fake_count = 0, 0
    random.shuffle(rows)
    for r in rows:
        ctx = r.get("context", "")
        label = r.get("pseudo_label")  # Get label directly from data

        if label == "fake":
            y = 0  # Fake = False = 0
            fake_count += 1
        elif label == "real":
            y = 1  # Real = True = 1
            real_count += 1
        else:
            continue  # Skip if label is unknown

        examples.append({"text": ctx, "label": y})
    return examples, real_count, fake_count


def compute_metrics(p):
    """Compute accuracy metric."""
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    return {
        "accuracy": sklearn.metrics.accuracy_score(labels, preds),
        "macro f1": sklearn.metrics.f1_score(labels, preds, average="macro"),
        "weighted f1": sklearn.metrics.f1_score(labels, preds, average="weighted"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="data/reviews_with_labels.jsonl")
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--output_dir", type=str, default="models/roberta_fake")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max_rows", type=int, default=None)  # For testing, adjust as needed
    args = parser.parse_args()

    # Load and prepare data
    print(f"[INFO] loading labeled data from {args.train_file}")
    raw_data = load_jsonl(args.train_file, max_rows=args.max_rows)
    examples, real_count, fake_count = prepare_examples(raw_data)
    print(f"[INFO] prepared {len(examples)} examples for training.")

    # Split into texts and labels
    texts = [e["text"] for e in examples]
    labels = [e["label"] for e in examples]
    dataset = Dataset.from_dict({"text": texts, "label": labels})

    # Split into train and test datasets
    train_test = dataset.train_test_split(test_size=0.1)
    # train_test = dataset.train_test_split(test_size=0.1, seed=42)

    # Tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)


    def tokenize_fn(ex):
        return tokenizer(ex["text"], truncation=True, padding="max_length", max_length=512)


    # Apply tokenization
    train_test = train_test.map(lambda ex: tokenize_fn(ex), batched=True)
    train_test.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Model setup
    model = RobertaForSequenceClassification.from_pretrained(args.model_name, num_labels=len(set(labels)))

    # set a batch size for compute_loss_func
    batch_size = 8

    # Training arguments setup
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=len(dataset) // 20,  # Number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # Strength of weight decay
        num_train_epochs=args.epochs,
        eval_strategy="epoch",  # Evaluate after each epoch
        logging_dir='./logs',  # Directory for storing logs
        load_best_model_at_end=True,  # Load the best model when finished training
        save_strategy="epoch",  # Save model after each epoch
        logging_strategy="epoch",  # Log after each epoch
        fp16=True,  # Enable mixed precision
        max_grad_norm=1.0,  # Gradient clipping
        gradient_accumulation_steps=2,  # Accumulate gradients to handle memory issues
        report_to="none",  # 禁用日志报告
        learning_rate=5e-5,  # Learning rate
        lr_scheduler_type="linear",  # Learning rate scheduler type (default: linear)
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_test["train"],
        eval_dataset=train_test["test"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # 3轮没有提升则停止训练
    )

    # Train the model
    trainer.train()

    # Save model and tokenizer
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"[INFO] Training complete. Model saved to {args.output_dir}")
