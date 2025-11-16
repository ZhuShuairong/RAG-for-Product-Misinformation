import argparse
import json
import re
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics import classification_report


# ------------------------------
# Load test data
# ------------------------------
def load_examples(file_path, max_rows=None):
    examples = []
    with open(file_path, "r", encoding="utf-8") as fin:
        total_lines = sum(1 for _ in fin)

    with open(file_path, "r", encoding="utf-8") as fin:
        for i, line in enumerate(tqdm(fin, total=total_lines, desc="Loading test data")):
            if max_rows and i >= max_rows:
                break
            j = json.loads(line)

            context = j.get("context", "")
            label = j.get("pseudo_label")  # GT label

            if label not in ("fake", "real"):
                continue

            examples.append((context, label))
    return examples


# ------------------------------
# Extract “label: fake/real” from model output
# ------------------------------
def parse_label(output_text):
    """
    从 T5 生成的序列中解析 label: xxx
    """
    output_text = output_text.lower()
    match = re.search(r"label:\s*(fake|real)", output_text)
    if match:
        return match.group(1)
    return None


# ------------------------------
# Evaluate explainer model
# ------------------------------
def evaluate_model(model_dir, test_file, max_rows=None, max_new_tokens=64):
    # Load dataset
    examples = load_examples(test_file, max_rows)

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    y_true, y_pred = [], []

    # Evaluation Loop
    pbar = tqdm(examples, desc="Evaluating", total=len(examples))

    for context, label in pbar:
        # Construct prompt
        prompt = (
            f"The review context: {context}\n"
            f"classify:"
        )

        # Tokenize
        inputs = tokenizer(
            prompt,
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        # Generate output
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                length_penalty=1.0,
                early_stopping=True
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Parse model predicted label
        pred_label = parse_label(decoded)
        if pred_label is None:
            pred_label = "fake"  # fallback to fake

        y_true.append(0 if label == "fake" else 1)
        y_pred.append(0 if pred_label == "fake" else 1)

        # 更新进度条显示准确率
        acc = sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)
        pbar.set_postfix(acc=f"{acc:.4f}")

    # 输出分类指标
    print("\n===== Classification Report =====")
    print(classification_report(
        y_true,
        y_pred,
        digits=4,
        zero_division=0
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory of trained T5 explainer model")
    parser.add_argument("--test_file", type=str, required=True,
                        help="JSONL test file")
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    evaluate_model(
        args.model_dir,
        args.test_file,
        args.max_rows,
        args.max_new_tokens
    )
