import argparse
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import classification_report
from tqdm import tqdm


def load_examples(file_path, max_rows=None):
    """加载测试数据并准备为模型输入格式"""
    exs = []
    with open(file_path, "r", encoding="utf-8") as fin:
        total_lines = sum(1 for _ in fin)
    with open(file_path, "r", encoding="utf-8") as fin:
        for i, line in enumerate(tqdm(fin, total=total_lines, desc="Loading test data")):
            if max_rows and i >= max_rows:
                break
            j = json.loads(line)
            ctx = j.get("context", "")
            label = j.get("pseudo_label")  # 使用 "pseudo_label" 作为评估标签
            if label == "fake":
                y = 0  # Fake = False = 0
            else:
                y = 1  # Real = True = 1
            exs.append((ctx, y))  # 使用评论内容和标签
    return exs


def evaluate_model(model_dir, test_file, max_rows=None):
    """评估模型性能"""
    # 加载数据
    exs = load_examples(test_file, max_rows)

    # 加载tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    # 将模型转移到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    y_true, y_pred = [], []

    # 创建进度条
    pbar = tqdm(exs, total=len(exs), desc="Evaluating")

    # 遍历所有测试样本
    for text, label in pbar:
        # 对文本进行tokenization
        inputs = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 在不计算梯度的情况下，进行前向传播
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits
            pred = logits.argmax(dim=-1).cpu().item()  # 获取预测的类别

        # 将预测结果和真实标签添加到列表
        y_true.append(label)
        y_pred.append(pred)

        # 更新进度条描述
        pbar.set_postfix(
            accuracy=f"{(sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / len(y_true)):.4f}",
        )

    # 输出评估报告
    print(classification_report(y_true, y_pred, digits=4))


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Directory of the trained model")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test data (JSONL file)")
    parser.add_argument("--max_rows", type=int, default=None, help="Maximum rows to evaluate (default: None for all rows)")
    args = parser.parse_args()

    # 调用评估函数
    evaluate_model(args.model_dir, args.test_file, args.max_rows)
