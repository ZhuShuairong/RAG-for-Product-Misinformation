import argparse
import json
import random
from pathlib import Path
from tqdm import tqdm


def load_jsonl(file_path):
    """Load the JSONL file."""
    records = []
    with open(file_path, "r", encoding="utf-8") as fin:
        total_lines = sum(1 for _ in fin)
    with open(file_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, total=total_lines, desc="Loading data"):
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(file_path, records):
    """Save the list of records as a JSONL file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def create_small_dataset(input_file, output_file, ratio_real_to_fake=1, test_ratio=0.2):

    if test_ratio > 1.0 or test_ratio < 0.0:
        raise ValueError("test_ratio must be between 0.0 and 1.0")

    # Load the full dataset
    records = load_jsonl(input_file)

    # Separate real and fake reviews
    real_reviews = [r for r in records if r.get("pseudo_label") == "real"]
    fake_reviews = [r for r in records if r.get("pseudo_label") == "fake"]

    if not fake_reviews:
        raise ValueError("No fake reviews found in the dataset.")
    if not real_reviews:
        raise ValueError("No real reviews found in the dataset.")

    # Randomly sample fake_reviews
    fake_sample_count = max(int(len(fake_reviews) * ratio_real_to_fake), 1)  # Twice the number of fake reviews
    sampled_real_reviews = random.sample(real_reviews, fake_sample_count)

    # Randomly sample test sets
    real_test_count = max(int(len(sampled_real_reviews) * test_ratio), 1)
    sampled_real_reviews_train, sampled_real_reviews_test = sampled_real_reviews[:-real_test_count], sampled_real_reviews[-real_test_count:]
    fake_test_count = max(int(len(fake_reviews) * test_ratio), 1)
    fake_reviews_train, fake_reviews_test = fake_reviews[:-fake_test_count], fake_reviews[-fake_test_count:]
    small_dataset_train = sampled_real_reviews_train + fake_reviews_train
    random.shuffle(small_dataset_train)
    small_dataset_test = sampled_real_reviews_test + fake_reviews_test
    random.shuffle(small_dataset_test)

    # Combine the sampled real reviews with all fake reviews
    small_dataset = sampled_real_reviews + fake_reviews

    # Shuffle the combined dataset
    random.shuffle(small_dataset)

    # Save the new small dataset
    save_jsonl(output_file, small_dataset)
    print(f"[INFO] Small dataset saved to {output_file} with {len(small_dataset)} records.")
    train_output_file = str(Path(output_file).with_name(Path(output_file).stem + "_train.jsonl"))
    save_jsonl(train_output_file, small_dataset_train)
    print(f"[INFO] Training set saved to {train_output_file} with {len(small_dataset_train)} records.")
    test_output_file = str(Path(output_file).with_name(Path(output_file).stem + "_test.jsonl"))
    save_jsonl(test_output_file, small_dataset_test)
    print(f"[INFO] Test set saved to {test_output_file} with {len(small_dataset_test)} records.")


if __name__ == "__main__":
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Generate a small dataset of real and fake reviews.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input reviews JSONL file.",default="data/reviews_with_labels.jsonl")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the generated small dataset JSONL file.",default="data/reviews_with_labels_small.jsonl")
    parser.add_argument("--ratio_real_to_fake", type=float, default=1.0, help="Ratio of real to fake reviews in the small dataset.")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Proportion of the dataset to include in the test split.")

    # Parse the arguments
    args = parser.parse_args()

    # Call the function to create a small dataset
    create_small_dataset(
        input_file=args.input_file,
        output_file=args.output_file,
        ratio_real_to_fake=args.ratio_real_to_fake,
        test_ratio=args.test_ratio
    )
