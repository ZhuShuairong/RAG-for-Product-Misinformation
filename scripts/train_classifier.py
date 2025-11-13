import argparse
import json
import os
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import random
from sklearn.model_selection import train_test_split

def load_jsonl(file_path, max_rows=None):
    rows = []
    with open(file_path, "r", encoding="utf-8") as fin:
        for i, line in enumerate(fin):
            if max_rows and i >= max_rows:
                break
            rows.append(json.loads(line))
    return rows

def prepare_examples(rows, max_retrieved_chars=400):
    exs = []
    for r in rows:
        ctx = r.get("context","")
        # join retrieved docs if exist
        retrieved = r.get("retrieved", [])
        retrieved_text = " ".join([d.get("document","")[:max_retrieved_chars] for d in retrieved])
        text = ctx + " || Retrieved: " + retrieved_text
        label = r.get("pseudo_label_v2") or r.get("pseudo_label") or "unknown"
        # map to numeric classes: fake/factual_error -> 1, real/consistent -> 0, unknown/suspicious -> 2
        if label in ["fake", "factual_error"]:
            y = 1
        elif label in ["real", "consistent"]:
            y = 0
        else:
            y = 2
        exs.append({"text": text, "label": y})
    return exs

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    acc = (preds == labels).mean()
    return {"accuracy": acc}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", type=str, default="data/reviews_with_labels.jsonl")
    p.add_argument("--model_name", type=str, default="roberta-base")
    p.add_argument("--output_dir", type=str, default="models/roberta_fake")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max_rows", type=int, default=20000)  # for testing, adjust as needed
    args = p.parse_args()

    raw = load_jsonl(args.train_file, max_rows=args.max_rows)
    print("[INFO] loaded", len(raw))
    examples = prepare_examples(raw)
    # filter out class 2 if small or include as separate class
    texts = [e["text"] for e in examples]
    labels = [e["label"] for e in examples]
    dataset = Dataset.from_dict({"text": texts, "label": labels})
    train_test = dataset.train_test_split(test_size=0.1, seed=42)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    def tokenize_fn(ex):
        return tokenizer(ex["text"], truncation=True, padding="max_length", max_length=256)
    train_test = train_test.map(lambda ex: tokenize_fn(ex), batched=True)
    train_test.set_format(type="torch", columns=["input_ids","attention_mask","label"])
    num_labels = len(set(labels))
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=args.epochs,
        # evaluation_strategy="epoch",  # The parameter `evaluation_strategy` has been renamed to `eval_strategy`.
        eval_strategy="epoch",  # 在每个 epoch 后进行评估
        save_strategy="epoch",  # 在每个 epoch 后保存模型
        logging_strategy="epoch",  # 在每个 epoch 后记录日志
        # logging_steps=100,  # 每 100 步记录一次日志
        # fp16=False,
        fp16=True,  # 启用混合精度训练
        max_grad_norm=1.0,  # 梯度裁剪
        gradient_accumulation_steps=2,  # 在显存不足时使用梯度累积
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_test["train"],
        eval_dataset=train_test["test"],
        compute_metrics=compute_metrics
    )
    trainer.train()

    # 保存模型和 tokenizer
    model.save_pretrained(args.output_dir)  # 保存训练的模型
    tokenizer.save_pretrained(args.output_dir)  # 保存tokenizer，包括vocab.json等文件

    print("[INFO] training done. model saved to", args.output_dir)
