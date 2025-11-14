import argparse
import json
import random
from pathlib import Path
import copy
from tqdm import tqdm
import ollama  # Local LLM client for Ollama model
import chromadb
from chromadb.config import Settings


# ------------------------------
# Helpers for JSONL I/O
# ------------------------------

def load_jsonl(path):
    """Load jsonl file"""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="Loading jsonl"):
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def save_jsonl(path, records):
    """Save list of dict to jsonl"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ------------------------------
# Helpers for extracting fields
# ------------------------------

def get_field(rec, key, default=None):
    """Get field from rec or rec["meta"]"""
    if key in rec and rec[key] not in (None, ""):
        return rec[key]
    meta = rec.get("meta", {})
    if isinstance(meta, dict) and key in meta and meta[key] not in (None, ""):
        return meta[key]
    return default


def extract_review_text(rec):
    """Extract review_text string"""
    txt = get_field(rec, "review_text", None)
    if txt:
        return str(txt)
    # fallback to context
    ctx = rec.get("context", "")
    return str(ctx)


def extract_product_name(rec, fallback="a skincare product"):
    """Get product name"""
    name = get_field(rec, "product_name", None)
    if name:
        return str(name)
    return fallback


# ------------------------------
# Split real/fake from labeled data
# ------------------------------

def split_real_fake(records, label_field="pseudo_label"):
    """Split records by label"""
    real, fake = [], []
    for r in records:
        label = r.get(label_field) or r.get("pseudo_label") or r.get("label")
        if label in ("real", 0, "0"):
            real.append(r)
        elif label in ("fake", 1, "1"):
            fake.append(r)
    return real, fake


# ------------------------------
# Product info retrieval from ChromaDB
# ------------------------------

class Retriever:
    def __init__(self, persist_dir="chroma_db", model_dir="./models/all-MiniLM-L6-v2"):
        # Load the model from the local directory
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_collection("products")

    def retrieve(self, query: str, top_k: int = 3):
        """
        Retrieve top_k relevant documents from ChromaDB based on the query.

        Args:
            query (str): The search query (e.g., product review or description).
            top_k (int): The number of top results to return.

        Returns:
            list: A list of dictionaries containing product information.
        """
        # Query ChromaDB for the top_k most relevant documents and their metadata
        res = self.collection.query(query_embeddings=[query], n_results=top_k, include=["documents", "metadatas"])

        # Process the response to extract the documents and their corresponding metadata
        docs = []
        for d, m, id_ in zip(res.get("documents", [[]])[0], res.get("metadatas", [[]])[0], res.get("ids", [[]])[0]):
            docs.append({
                "id": id_,
                "document": d,
                "metadata": m  # The metadata contains all product-related information
            })
        return docs


# ------------------------------
# Prompt + Ollama generation
# ------------------------------

def build_prompt(real_review_text, fake_style_text, product_meta):
    """Build English prompt with product information as a string"""
    real_review_text = real_review_text.replace("\n", " ").strip()
    fake_style_text = fake_style_text.replace("\n", " ").strip()

    # Convert product meta to a string (e.g., JSON string or formatted string)
    product_info_str = json.dumps(product_meta, indent=2)  # Convert product_meta to a readable string

    prompt = f"""
You are helping generate synthetic low-quality or fake-looking product reviews for a machine learning dataset.

Here is a REAL user review for the product:
---
{real_review_text}
---

Here is an EXAMPLE of a low-quality / fake-looking review style:
---
{fake_style_text}
---

Now write ONE NEW fake-looking review for the SAME product with the following details:
---
{product_info_str}
---

Requirements:
- Use exaggerated or promotional language.
- Use an extreme sentiment (either very positive or very negative).
- Keep it short to medium length (about 40–120 words).
- Do NOT mention that you are an AI or that this is synthetic or fake.
- Do NOT talk about writing reviews, datasets, or training.
- The text should sound like a normal user review on an e-commerce site.

Return ONLY the review text, nothing else.
"""
    return prompt.strip()


def generate_fake_review_with_ollama(real_rec, fake_rec, model_name, retriever):
    """Call Ollama to get one fake review"""
    real_text = extract_review_text(real_rec)
    fake_style_text = extract_review_text(fake_rec)

    # Retrieve product meta from ChromaDB based on real_rec
    product_meta = retriever.retrieve(real_text, top_k=1)[0]["metadata"]

    # Build prompt with product information and review text
    prompt = build_prompt(real_text, fake_style_text, product_meta)

    resp = ollama.chat(  # call local model
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    content = resp.get("message", {}).get("content", "").strip()
    return content or fake_style_text  # fallback to style text if empty


# ------------------------------
# Simple text augmentation for fake reviews (without Ollama)
# ------------------------------

def augment_fake_review(fake_review):
    """Augment the fake review using simple text manipulation"""
    # 1. Randomly replace words with their synonyms
    synonyms = {
        "good": ["amazing", "excellent", "outstanding"],
        "bad": ["terrible", "horrible", "awful"],
        "love": ["adore", "enjoy", "like"],
        "hate": ["dislike", "loathe", "despise"],
    }
    words = fake_review.split()
    augmented_words = []

    for word in words:
        lower_word = word.lower()
        if lower_word in synonyms:
            new_word = random.choice(synonyms[lower_word])
            augmented_words.append(new_word)
        else:
            augmented_words.append(word)

    # 2. Randomly shuffle some words to change sentence structure (very basic)
    if random.random() > 0.7:
        random.shuffle(augmented_words)

    return " ".join(augmented_words)


# ------------------------------
# Generate synthetic fake records
# ------------------------------

def generate_random_review_info():
    """Generate random review-related information"""
    # Rating: extreme 1 or 5
    rating = random.choice([1, 5])

    # Is recommended: random but consistent-ish
    is_rec = 1 if rating >= 4 else 0

    # Helpfulness + feedback counts: make them small/low-quality
    total_feedback = random.randint(0, 5)
    if total_feedback == 0:
        pos = 0
        neg = 0
        helpful = 0.0
    else:
        pos = random.randint(0, total_feedback)
        neg = total_feedback - pos
        helpful = pos / total_feedback if total_feedback > 0 else 0.0

    review_title = "Amazing product!" if rating == 5 else "Terrible experience"

    return {
        "rating": rating,
        "is_recommended": is_rec,
        "total_feedback_count": total_feedback,
        "total_pos_feedback_count": pos,
        "total_neg_feedback_count": neg,
        "helpfulness": helpful,
        "review_title": review_title
    }


def generate_synthetic_fake_records(real_reviews, fake_reviews, n_samples, model_name, retriever, ollama_combine_ratio=1, online_troll_behavior=False):
    """
    Generate many fake records (using only existing fake reviews)

    Args:
        real_reviews (list): List of real review records (dict).
        fake_reviews (list): List of fake review records (dict).
        n_samples (int): Number of synthetic fake samples to generate.
        model_name (str): Ollama model name to use.
        retriever (Retriever): Retriever instance for product info.
        ollama_combine_ratio (int): Ratio of combining existing fake reviews to Ollama generated reviews.
        online_troll_behavior (bool): If True, simulate online troll behavior by adding more duplicates.
    """
    if not real_reviews:
        raise ValueError("No real reviews found. Cannot generate synthetic data.")
    if not fake_reviews:
        raise ValueError(f"No fake reviews found. Cannot generate synthetic data.")

    synthetic_records = []

    # Number of fake reviews generated by combining fake reviews and Ollama generated fake reviews
    fake_reviews_needed = n_samples // ollama_combine_ratio

    print(f"[INFO] Generating {n_samples} fake reviews with Ollama and {fake_reviews_needed} fake reviews by combining with existing fake reviews.")

    for step in tqdm(range(n_samples), desc="Generating synthetic fake reviews"):
        # Generate fake review by randomly choosing real and fake samples
        real_rec = random.choice(real_reviews)
        fake_rec = random.choice(fake_reviews)

        # 1. Generate one fake review using Ollama
        ollama_fake = generate_fake_review_with_ollama(real_rec, fake_rec, model_name, retriever)

        # Create a new record based on real_rec skeleton (Ollama generated fake review)
        new_rec_ollama = copy.deepcopy(real_rec)
        new_rec_ollama["review_text"] = ollama_fake
        new_rec_ollama["pseudo_label"] = "fake"  # Use original label field

        # Generate random review-related information
        random_info = generate_random_review_info()
        new_rec_ollama.update(random_info)

        synthetic_records.append(new_rec_ollama)

        if online_troll_behavior and step % 1000 == 0:
            for _ in range(random.randint(0, 10)):
                new_rec_ollama = copy.deepcopy(real_rec)
                new_rec_ollama["review_text"] = ollama_fake
                new_rec_ollama["pseudo_label"] = "fake"  # Use original label field
                random_info = generate_random_review_info()
                new_rec_ollama.update(random_info)
                synthetic_records.append(new_rec_ollama)

        # 2. Combine existing fake reviews with the generated review for another fake review
        if step % ollama_combine_ratio == 0:
            # combined_fake = augment_fake_review(extract_review_text(fake_rec))
            combined_fake = augment_fake_review(extract_review_text(ollama_fake))

            new_rec_combined = copy.deepcopy(real_rec)
            new_rec_combined["review_text"] = combined_fake
            new_rec_combined["pseudo_label"] = "fake"  # Use original label field

            # Generate random review-related information
            random_info = generate_random_review_info()
            new_rec_combined.update(random_info)

            synthetic_records.append(new_rec_combined)

            if online_troll_behavior and step % 1000 == 0:
                for _ in range(random.randint(0, 100)):
                    combined_fake = augment_fake_review(extract_review_text(ollama_fake))
                    new_rec_combined = copy.deepcopy(real_rec)
                    new_rec_combined["review_text"] = combined_fake
                    new_rec_combined["pseudo_label"] = "fake"  # Use original label field
                    random_info = generate_random_review_info()
                    new_rec_combined.update(random_info)
                    synthetic_records.append(new_rec_combined)

    return synthetic_records


# ------------------------------
# Main: load -> generate -> save
# ------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic fake reviews with Ollama and save augmented train jsonl.")
    parser.add_argument("--in_file", type=str, required=True, help="Input labeled jsonl file (with real/fake labels).")
    parser.add_argument("--out_fake_file", type=str, required=True, help="Output jsonl file for synthetic fake reviews.")
    parser.add_argument("--out_train_file", type=str, required=True, help="Output jsonl file for augmented training data.", default="data/train_augmented.jsonl")
    parser.add_argument("--model", type=str, default="llama3.2:3b", help="Ollama model name to use.")
    parser.add_argument("--ollama_ratio", type=float, default=1.0, help="Ratio of Ollama generated fake reviews to original real reviews.")
    parser.add_argument("--ollama_combine_ratio", type=int, default=1, help="Ratio of combining existing fake reviews to Ollama generated reviews. E.g., 2 means for every 2 Ollama reviews, generate 1 combined fake review.")
    parser.add_argument("--online_troll_behavior", action="store_true", help="Simulate online troll behavior by adding more duplicate synthetic reviews.")
    args = parser.parse_args()

    in_path = Path(args.in_file)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    print(f"[INFO] loading labeled data from {in_path}")
    records = load_jsonl(in_path)

    # Initialize the Chroma retriever
    retriever = Retriever(model_dir="./models/all-MiniLM-L6-v2")

    real_reviews, fake_reviews = split_real_fake(records)
    real_count = len(real_reviews)
    fake_seed_count = len(fake_reviews)
    print(f"[INFO] real samples: {real_count}, seed fake samples: {fake_seed_count}")

    if real_count == 0:
        raise ValueError("No real samples found in input file.")

    n_synth = int(args.ollama_ratio * real_count)
    if n_synth <= 0:
        print("[INFO] ollama_combine_ratio <= 0, nothing to generate. Just copying input to out_train_file.")
        save_jsonl(args.out_train_file, records)
        raise ValueError(f"[Error] ollama_combine_ratio ({args.ollama_combine_ratio}) too small, no synthetic samples ({n_synth}) to generate.")

    print(f"[INFO] generating {n_synth} synthetic fake reviews with Ollama")
    synthetic_fakes = generate_synthetic_fake_records(
        real_reviews=real_reviews,
        fake_reviews=fake_reviews,
        n_samples=n_synth,
        model_name=args.model,
        retriever=retriever,
        ollama_combine_ratio=args.ollama_combine_ratio,
        online_troll_behavior=args.online_troll_behavior,
    )

    print(f"[INFO] saving synthetic fake reviews to {args.out_fake_file}")
    save_jsonl(args.out_fake_file, synthetic_fakes)

    # Merge original + synthetic, then shuffle
    all_train = records + synthetic_fakes
    random.shuffle(all_train)

    print(f"[INFO] saving augmented training data to {args.out_train_file}")
    save_jsonl(args.out_train_file, all_train)

    print("[INFO] done.")
