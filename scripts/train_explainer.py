import argparse
import json
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer,
                          EarlyStoppingCallback)
import torch
import random
import numpy as np
import sklearn
from tqdm import tqdm


# ----------------------------------------------------------
# Load JSONL
# ----------------------------------------------------------
def load_jsonl(file_path, max_rows=None):
    rows = []
    with open(file_path, "r", encoding="utf-8") as fin:
        total_lines = sum(1 for _ in fin)
    with open(file_path, "r", encoding="utf-8") as fin:
        for i, line in enumerate(tqdm(fin, total=total_lines, desc="Loading data")):
            if max_rows and i >= max_rows:
                break
            rows.append(json.loads(line))
    return rows


# ----------------------------------------------------------
# Prepare training examples
# (review context → label + reason)
# ----------------------------------------------------------
def prepare_examples(rows):
    examples = []
    random.shuffle(rows)

    for r in rows:
        context = r.get("context", "")
        label = r.get("pseudo_label", None)

        if label not in ("fake", "real"):
            continue

        # --- template reasons ---
        if label == "fake":
            reason = "The product information contradicts the review, or the review content is too vague or inconsistent."
        else:
            reason = "The review content matches the product features and provides meaningful detail."

        examples.append({
            "context": context,
            "label": label,
            "reason": reason
        })

    return examples


# ----------------------------------------------------------
# Tokenizer function
# ----------------------------------------------------------
def preprocess_fn(batch, tokenizer, max_input_len=512, max_output_len=128):
    inputs = []
    targets = []

    for context, label, reason in zip(
            batch["context"], batch["label"], batch["reason"]
    ):
        # Construct prompt
        input_text = (
            f"The review context: {context}\n"
            f"classify:"
        )
        target_text = f"label: {label}. reason: {reason}"

        inputs.append(input_text)
        targets.append(target_text)

    model_inputs = tokenizer(
        inputs,
        max_length=max_input_len,
        truncation=True,
        padding="max_length",
    )

    labels = tokenizer(
        targets,
        max_length=max_output_len,
        truncation=True,
        padding="max_length",
    )

    # Replace padding token IDs with -100
    label_ids = labels["input_ids"]
    label_ids = [
        [(lid if lid != tokenizer.pad_token_id else -100) for lid in label_row]
        for label_row in label_ids
    ]

    model_inputs["labels"] = label_ids
    return model_inputs


# ----------------------------------------------------------
# Compute metrics
# ----------------------------------------------------------
def compute_metrics(p):
    """Compute accuracy and F1 score."""
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    return {
        "accuracy": sklearn.metrics.accuracy_score(labels, preds),
        "macro f1": sklearn.metrics.f1_score(labels, preds, average="macro"),
        "weighted f1": sklearn.metrics.f1_score(labels, preds, average="weighted"),
    }


# ----------------------------------------------------------
# Trainer
# ----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="data/reviews_with_labels.jsonl")
    parser.add_argument("--model_name", type=str, default="t5-base")
    parser.add_argument("--output_dir", type=str, default="models/fake_explainer")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max_rows", type=int, default=None)
    args = parser.parse_args()

    print(f"[INFO] loading data from {args.train_file}")
    raw_data = load_jsonl(args.train_file, max_rows=args.max_rows)

    # build examples
    examples = prepare_examples(raw_data)
    print(f"[INFO] Prepared {len(examples)} examples")

    # build HF dataset
    dataset = Dataset.from_dict({
        "context": [e["context"] for e in examples],
        "label": [e["label"] for e in examples],
        "reason": [e["reason"] for e in examples],
    })

    dataset = dataset.train_test_split(test_size=0.1)

    # tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    lora_t5_flag = False  # mark loRA T5 model
    model, tokenizer = None, None
    if args.model_name in ["t5-base", "lora-t5", "t5-small", "lora-t5-small"]:
        if args.model_name in ["lora-t5", "lora-t5-small"]:
            if args.model_name == "lora-t5":
                args.model_name = "t5-base"  # Use base model for loading
            elif args.model_name == "lora-t5-small":
                args.model_name = "t5-small"  # Use base model for loading
            lora_t5_flag = True
        from transformers import T5Tokenizer, T5ForConditionalGeneration

        tokenizer = T5Tokenizer.from_pretrained(args.model_name)
        model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    elif args.model_name in ["facebook/bart-large", "facebook/bart-base"]:
        from transformers import BartTokenizer, BartForConditionalGeneration

        tokenizer = BartTokenizer.from_pretrained(args.model_name)
        model = BartForConditionalGeneration.from_pretrained(args.model_name)
    else:
        model, tokenizer = None, None
    if model is None or tokenizer is None:
        raise ValueError(f"Unsupported model_name: {args.model_name}")

    if lora_t5_flag:
        from peft import PeftModel, LoraConfig

        model = PeftModel(
            model=model,
            peft_config=LoraConfig(
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
            ),
        )

    # map preprocess
    dataset = dataset.map(
        lambda batch: preprocess_fn(batch, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
        keep_in_memory=False  # Avoid keeping data in memory to manage memory consumption
    )

    dataset.set_format(type="torch")

    # define batch size
    batch_size = 1

    # training settings
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=5e-5,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True,
        gradient_accumulation_steps=2,
        warmup_ratio=0.05,
        save_total_limit=1,  # Keep only the best model
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,  # Add the metrics function here
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Stop training if no improvement for 2 evals
    )

    print("[INFO] Starting training...")
    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[INFO] Training finished. Model saved to {args.output_dir}")
