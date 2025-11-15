import json
import random


def get_reviews_to_test(file_path, real_num=1, fake_num=1):
    with open(file_path, "r", encoding="utf-8") as fin:
        total_lines = sum(1 for _ in fin)
    test_reviews = []
    real_count, fake_count = 0, 0
    while real_count < real_num or fake_count < fake_num:
        with open(file_path, "r", encoding="utf-8") as fin:
            for _ in range(random.randint(0, total_lines)):
                line = fin.readline()
            line_json = json.loads(line)
            context, pseudo_label = line_json.get('context'), line_json.get('pseudo_label')
            if pseudo_label == 'real' and real_count < real_num:
                test_reviews.append([context, pseudo_label, line_json])
                real_count += 1
            if pseudo_label == 'fake' and fake_count < fake_num:
                test_reviews.append([context, pseudo_label, line_json])
                fake_count += 1
    return test_reviews

def print_test_reviews(test_reviews):
    for context, pseudo_label, full_json in test_reviews:
        print("Context:", context)
        print("Pseudo Label:", pseudo_label)
        # print("Full JSON:", json.dumps(full_json, ensure_ascii=False, indent=2))
        print("-" * 80)

test_reviews = get_reviews_to_test("data/reviews_with_labels_small.jsonl", real_num=1, fake_num=1)
print_test_reviews(test_reviews)
