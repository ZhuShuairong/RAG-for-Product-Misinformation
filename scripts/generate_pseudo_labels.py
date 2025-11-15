import json
import argparse
from build_retriever import Retriever
from tqdm import tqdm
import re


def safe_float(x, default=None):
    """安全转换为 float"""
    try:
        return float(x)
    except (ValueError, TypeError):
        return default


def safe_int(x, default=None):
    """安全转换为 int"""
    try:
        return int(x)
    except (ValueError, TypeError):
        return default


def extract_capacity_from_text(text):
    """从评论文本中提取所有容量信息（如 ml, oz, g 等）"""
    # 正则表达式匹配容量，单位可以是 ml, oz, g 等
    pattern = r'(\d+(\.\d+)?)\s?(ml|oz|g|lbs|kg|mL|fl\s?oz|oz\.)'
    matches = re.findall(pattern, text.lower())  # 提取所有匹配项

    # 如果有匹配项，返回每个容量和单位的组合
    capacities = []
    for match in matches:
        amount = float(match[0])  # 提取容量
        unit = match[2]  # 提取单位
        capacities.append([amount, unit])  # 返回为 [容量, 单位] 的列表

    return capacities


def extract_price_from_text(text):
    """从评论文本中提取价格信息（如 $10.99, 20 dollars, etc）"""
    # pattern = r'(\$?\d+(\.\d{1,2})?)\s?(usd|dollars|\$)?'  # 支持不同价格形式，如 $10.99 或 20 dollars
    # 支持不同价格形式，如 $10.99 或 20 dollars 或 spend / cost / paid / pay 5
    pattern = r'(?:spend|cost|paid|pay)?\s*\$?(\d+(\.\d{1,2})?)\s?(usd|dollar(s)?|\$)?'  # BUG: 会匹配所有数字
    matches = re.findall(pattern, text.lower())
    if matches:
        # 返回提取的价格
        return safe_float(matches[0][0].replace('$', ''))
    return None


def convert_to_ml(amount, unit):
    """将容量转换为 ml（如果单位是 ml、oz、g 等）"""
    conversion_factors = {
        "ml": 1,
        "M": 1,
        "oz": 29.5735,  # 1 oz = 29.5735 ml
        "g": 1,  # 假设 g 和 ml 是等价的（对于水等物质），对于其他物质可以进行更精确的转换
        "lbs": 453.592,  # 1 lb = 453.592 g
        "kg": 1000  # 1 kg = 1000 g
    }
    if unit in conversion_factors:
        return amount * conversion_factors[unit]
    return amount


def detect_skin_type_mismatch(text, product_meta):
    """检测评论中的皮肤类型与产品描述之间的矛盾"""
    skin_type_keywords = ["oily", "dry", "combination", "sensitive"]
    gender_keywords = ["men", "women", "unisex", "gender-neutral", "genderless"]

    # 适用皮肤类型矛盾
    if any(keyword in text for keyword in skin_type_keywords):  # 检查评论中提到的皮肤类型
        if any(skin_type in product_meta.get("highlights", "") for skin_type in skin_type_keywords):  # 避免不存在皮肤类型信息的情况
            if (
                    "oily" in text
                    and "dry" in product_meta.get("highlights", "")
            ):
                return "fake"  # 例如，评论说适合油性皮肤，产品标称适合干性皮肤
            if (
                    "dry" in text
                    and "oily" in product_meta.get("highlights", "")
            ):
                return "fake"  # 例如，评论说适合干性皮肤，产品标称适合油性皮肤

    # 适用性别矛盾
    if any(gender in text for gender in gender_keywords):  # 检查评论中提到的性别
        if any(gender in product_meta.get("highlights", "") for gender in gender_keywords):  # 避免不存在性别信息的情况
            if (
                    "men" in text
                    and "unisex" not in product_meta.get("highlights", "")
                    and "genderless" not in product_meta.get("highlights", "")
            ):
                return "fake"  # 如果评论说适合男士，但产品标称适合所有性别
            if (
                    "women" in text
                    and "unisex" not in product_meta.get("highlights", "")
                    and "genderless" not in product_meta.get("highlights", "")
            ):
                return "fake"  # 如果评论说适合女士，但产品标称适合所有性别

    return "real"


def compute_fake_score(j, review_context, retrieved_docs):
    """计算评论的假评论启发式得分"""
    ctx = j.get("context", "") or ""
    meta = j.get("meta", {}) or {}

    # 提取字段并确保其安全
    rating = safe_float(meta.get("rating"), None)
    is_recommended = safe_int(meta.get("is_recommended"), None)
    helpfulness = safe_float(meta.get("helpfulness"), None)
    total_feedback_count = safe_int(meta.get("total_feedback_count"), None)
    total_neg_feedback_count = safe_int(meta.get("total_neg_feedback_count"), None)
    total_pos_feedback_count = safe_int(meta.get("total_pos_feedback_count"), None)

    text = (ctx).strip()
    text_len = len(text)

    fake_score = 0  # 假评论得分初始化

    # 启发式规则检测假评论
    if rating is not None and text:
        if rating in [1.0, 5.0] and text_len < 20:
            fake_score += 2
        # 计算负面反馈比例
        neg_total_percentage = (total_neg_feedback_count / total_feedback_count) if total_feedback_count and total_feedback_count > 0 else 0
        # 计算正面反馈比例
        pos_total_percentage = (total_pos_feedback_count / total_feedback_count) if total_feedback_count and total_feedback_count > 0 else 0
        # 低帮助率且反馈数量足够
        if helpfulness is not None and total_feedback_count is not None and total_feedback_count >= 3 and helpfulness < 0.3:
            fake_score += 1
            # 根据反馈数量调整负面反馈比例阈值
            if total_feedback_count >= 25:
                fake_score += 1 if neg_total_percentage > 0.6 else 0
            elif total_feedback_count >= 10:
                fake_score += 1 if neg_total_percentage > 0.7 else 0
            elif total_feedback_count >= 5:
                fake_score += 1 if neg_total_percentage > 0.8 else 0
            else:
                fake_score += 0
        # 低反馈数量且极端评分
        if total_feedback_count == 0 and rating in [1.0, 5.0] and text_len < 40:
            fake_score += 1
        # 评分与推荐状态不一致
        if rating <= 2 and (is_recommended == 1 or neg_total_percentage > 0.8):
            fake_score += 1  # 低评分但推荐
        if rating >= 4 and (is_recommended == 0 or neg_total_percentage > 0.8):
            fake_score += 1  # 高评分但不推荐
    elif rating is None and text:
        if text_len < 20:
            return float('inf')  # 返回高分标记为假（未知）
        elif text_len < 50:
            fake_score += 2
        else:
            fake_score += 1
    else:  # 缺少文本
        return float('inf')  # 返回高分标记为假（未知）

    return fake_score


def decide_final_label(fact_label, fake_score, j, fake_threshold=2):
    """决策最终的伪标签（真实 vs 虚假）"""
    ctx = j.get("context", "") or ""
    meta = j.get("meta", {}) or {}
    base_label = j.get("pseudo_label", None)

    # 提取评分
    rating = safe_float(meta.get("rating"), None)

    # 如果缺少必要字段，标记为未知
    if fact_label == "fake":
        return "fake"

    # 启发式假评论检测
    if fake_score > fake_threshold or base_label == "fake":
        return "fake"

    # 判断是否标记为真实
    if len(ctx) >= 60 and rating is not None:
        return "real"

    return "fake"  # 默认为假


def detect_factual_mismatch(review_context, retrieved_docs):
    """检测评论文本与检索到的商品文档之间的简单事实性矛盾"""
    text = review_context.lower()

    for doc in retrieved_docs:
        doc_text = doc["document"].lower()
        doc_meta = doc["metadata"]

        # 1. 适用范围矛盾（皮肤类型，适用对象）
        skin_type_mismatch = detect_skin_type_mismatch(text, doc_meta)
        if skin_type_mismatch == "fake":
            return "fake"

        # 2. 容量矛盾
        comment_capacity_ml_list = [convert_to_ml(comment_capacity, comment_unit) for comment_capacity, comment_unit in extract_capacity_from_text(text)]
        product_size = doc_meta.get("size", "").lower()
        product_capacity_ml_list = [convert_to_ml(prod_capacity, prod_unit) for prod_capacity, prod_unit in extract_capacity_from_text(product_size)]
        if comment_capacity_ml_list and product_capacity_ml_list:
            capacity_fake_flag = []
            for comment_capacity_ml in comment_capacity_ml_list:
                for product_capacity_ml in product_capacity_ml_list:
                    if abs(comment_capacity_ml - product_capacity_ml) > (product_capacity_ml * 0.2) and abs(comment_capacity_ml - product_capacity_ml) > 50:
                        capacity_fake_flag.append(True)  # 容量差异过大，继续检查其他容量
                    else:
                        capacity_fake_flag.append(False)  # 只要有一个容量匹配，就认为容量信息是真实的
            if all(capacity_fake_flag):  # 所有容量均不匹配
                return "fake"  # 容量差异过大，标记为假

        # # 3. 价格矛盾  # BUG: 会匹配所有数字
        # comment_price = extract_price_from_text(text)
        # if comment_price:
        #     price_usd = safe_float(doc_meta.get("price_usd", 0))
        #     value_price_usd = safe_float(doc_meta.get("value_price_usd", 0))
        #     sale_price_usd = safe_float(doc_meta.get("sale_price_usd", 0))
        #     print(text, comment_price, price_usd, value_price_usd, sale_price_usd)
        #     sage_percentage = 0.1  # 允许的价格误差百分比，默认为0.2
        #     price_fake_flag = []
        #     for prod_price in [price_usd, value_price_usd, sale_price_usd]:
        #         if prod_price and abs(comment_price - prod_price) > (prod_price * sage_percentage):
        #             price_fake_flag.append(True)  # 价格差异过大，继续检查其他价格
        #         else:
        #             price_fake_flag.append(False)  # 只要有一个价格匹配，就认为价格信息是真实的
        #     if all(price_fake_flag):  # 所有价格均不匹配
        #         return "fake"  # 价格差异过大，标记为假

        # 其他检测逻辑...

        # 线下购买与仅支持线上购买矛盾
        online_only = doc_meta.get("online_only")
        if online_only:
            if re.search(r'\b(bought in store|purchased at store|found in store|seen in store|available in store|went to the store|visited the store|in-store purchase|brick and mortar)\b', text):
                return "fake"

    return "real"


def detect_duplicate_reviews(reviews):
    """检测重复评论并标记为 fake"""
    import hashlib  # 导入 hashlib 用于计算哈希值

    seen = set()
    duplicates = []  # 用于存储重复评论

    for review in tqdm(reviews, desc="Detecting duplicate reviews"):
        # 计算评论文本的 MD5 哈希值
        review_text = review.get("context", "")
        review_hash = hashlib.sha3_256(review_text.encode('utf-8')).hexdigest()  # 由于评论可能较长，使用哈希值进行比较

        # 如果已经出现过该评论（哈希值重复），则为重复评论
        if review_hash in seen:
            review["pseudo_label"] = "fake"  # 标记为 fake
            duplicates.append(review)
        else:
            seen.add(review_hash)

    return reviews, duplicates


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, default="data/reviews_context.jsonl")
    parser.add_argument("--out_file", type=str, default="data/reviews_with_labels.jsonl")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    retriever = Retriever(model_dir="models/all-MiniLM-L6-v2")  # 初始化检索器

    # 统计总行数以便进度条显示
    with open(args.in_file, "r", encoding="utf-8") as fin:
        total_lines = sum(1 for _ in fin)

    total_count = 0
    real_count = 0
    fake_count = 0

    reviews = []

    with open(args.in_file, "r", encoding="utf-8") as fin, open(args.out_file, "w", encoding="utf-8") as fout:
        pbar = tqdm(fin, total=total_lines, desc="Processing reviews")

        for line in pbar:
            j = json.loads(line)
            context_text = j.get("context", "")
            base_label = j.get("pseudo_label", None)

            # RAG 事实一致性检测
            retrieved = retriever.retrieve(context_text, top_k=args.top_k)
            fact_label = detect_factual_mismatch(context_text, retrieved)

            # 假评论启发式评分
            fake_score = compute_fake_score(j, context_text, retrieved)

            # 生成最终标签
            final_label = decide_final_label(
                fact_label,  # 事实一致性标签
                fake_score,  # 假评论得分
                j,  # 原始评论数据
                fake_threshold=2  # 假评论得分阈值, fake_score >= fake_threshold 则标记为假
            )

            if final_label == "fake":
                fake_count += 1
            else:
                real_count += 1

            j["retrieved"] = retrieved  # record retrieved documents  # 记录检索到的文档
            # j["factual_consistency"] = fact_label  # 记录事实一致性标签
            # j["fake_score"] = fake_score  # 记录假评论得分
            j["pseudo_label"] = final_label  # 更新伪标签为最终标签

            fout.write(json.dumps(j, ensure_ascii=False) + "\n")
            total_count += 1
            reviews.append(j)

            pbar.set_postfix(
                # fact_label=fact_label,
                # current_fake_score=fake_score,
                # final_label=final_label,
                real_count=real_count,
                fake_count=fake_count,
                total_count=total_count
            )

    # TODO: 可选的重复评论处理（代码未经过测试验证）
    # # 处理重复评论
    # reviews, duplicates = detect_duplicate_reviews(reviews)
    #
    # # 将重复评论写入输出文件
    # with open(args.out_file.replace(".jsonl", "_cleaned.jsonl"), "w", encoding="utf-8") as fout:
    #     for review in reviews:
    #         fout.write(json.dumps(review, ensure_ascii=False) + "\n")
    # with open(args.out_file.replace(".jsonl", "_duplicates.jsonl"), "w", encoding="utf-8") as fout:
    #     for dup in duplicates:
    #         fout.write(json.dumps(dup, ensure_ascii=False) + "\n")

    print("[INFO] finished:", total_count)
    print("[STATS] real:", real_count, "fake:", fake_count, "total:", total_count)
