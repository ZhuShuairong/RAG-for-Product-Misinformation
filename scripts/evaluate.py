#!/usr/bin/env python3
"""
evaluate.py
Usage:
    python scripts/evaluate.py --model_dir models/roberta_fake --test_file data/reviews_with_labels.jsonl
"""
import argparse
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import classification_report
from tqdm import tqdm

def load_examples(file_path, max_rows=None):
    exs = []
    with open(file_path, "r", encoding="utf-8") as fin:
        for i, line in enumerate(fin):
            if max_rows and i >= max_rows:
                break
            j = json.loads(line)
            ctx = j.get("context","")
            retrieved_text = " ".join([d.get("document","")[:400] for d in j.get("retrieved", [])])
            text = ctx + " || Retrieved: " + retrieved_text
            label = j.get("pseudo_label_v2")  # Using pseudo_label_v2 for evaluation
            if label == "fake":
                y = 1
            else:
                y = 0  # real label
            exs.append((text, y))
    return exs


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--test_file", type=str, default="data/reviews_with_labels.jsonl")
    p.add_argument("--max_rows", type=int, default=2000)
    args = p.parse_args()

    exs = load_examples(args.test_file, args.max_rows)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    y_true, y_pred = [], []
    for text, label in tqdm(exs):
        inputs = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt")
        inputs = {k:v.to(device) for k,v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits
            pred = logits.argmax(dim=-1).cpu().item()
        y_true.append(label)
        y_pred.append(pred)
    print(classification_report(y_true, y_pred, digits=4))
