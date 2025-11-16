# RAG Fake Review Misinformation

This project implements a Retrieval Augmentation (RAG)-based model for detecting fake reviews and generating explanations based on product reviews. It utilizes Transformer models (such as RoBERTa) to classify reviews, or leverages Transformer models (such as T5 and BART) to generate classification results and explanations based on context and product information.

---

## Project Structure

```
RAG_FakeReview/
├── data/
│ ├── reviews_with_labels_small_train.jsonl # Training dataset for reviews with labels
│ ├── reviews_with_labels_small_test.jsonl # Test dataset
│ ├── reviews_context.jsonl # Training dataset for reviews with additional context
│ ├── product_info.csv # Product information CSV file
│ └── reviews_*.csv # Raw review datasets
├── scripts/
│ ├── download_sephora.py # Script to download Sephora review dataset
│ ├── build_retrieval.py # Script to build the retrieval database from product info
│ ├── train_classifier.py # Script for training the review classification model
│ ├── train_explainer.py # Script for training the explainer model and generating causes
│ ├── evaluate_classifier.py # Script for evaluating the classification model
│ ├── evaluate_explainer.py # Script for evaluating the explainer model
│ ├── generate_pseudo_labels.py # Script for generating pseudo-labels for reviews
│ ├── generate_synthetic_fakes.py # Script for generating synthetic fake reviews
│ ├── insert_products.py # Insert product information into the database
│ ├── retrieve_example.py # Example script for testing retrieval-based models
│ ├── prepare_context.py # Prepare context and review data for training
│ └── create_small_dataset.py # Create a small dataset for testing
├── qt5_classifier.py # Lightweight classifier using Qt5
├── qt5_explainer.py # Lightweight explainer using Qt5
├── create_reviews_to_test.py # To get random reviews for testing
├── models/ # Folder for storing pre-trained and fine-tuned models
├── chroma_db/ # Folder for storing ChromaDB database files
├── requirements.txt # Project dependencies
└── README.md # Project documentation (this file)
````

---

## Installation

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
````

---

## Downloading Dataset

Download the Sephora review dataset:

```bash
python scripts/download_sephora.py --dataset nadyinky/sephora-products-and-skincare-reviews --out_dir data
```

---

## Prepare context for reviews

Prepare the context for reviews by merging product information:

```bash
python scripts/prepare_context.py --input_dir data --out_file data/reviews_context.jsonl
```

---

## Insert product information into vector database

```bash
python scripts/insert_products.py --products_csv data/product_info.csv --persist_dir chroma_db --batch_size 256
```

---

### Generating Pseudo Labels

Generate pseudo labels for the comments:

```bash
python scripts/generate_pseudo_labels.py --in_file data/reviews_context.jsonl --out_file data/reviews_with_labels.jsonl --top_k 3
```

---

### Create a smaller dataset for quick experimentation

```bash
python scripts/create_small_dataset.py --input_file data/reviews_with_labels.jsonl --output_file data/reviews_with_labels_small.jsonl --ratio_real_to_fake 1.0 --split --test_ratio 0.1
```

---

### Generating Synthetic Fake Reviews (Optional)

Generating Synthetic Fake Reviews:

```bash
python scripts/generate_synthetic_fakes.py --in_file data/reviews_with_labels.jsonl --out_fake_file data/synthetic_fake_reviews.jsonl --out_train_file data/train_augmented.jsonl --model llama3.2:3b --ollama_ratio 0.3 --ollama_combine_ratio 1 --online_troll_behavior
```

---

## Usage

### Training the Classifier

To train a review classification model (true/false):

```bash
python scripts/train_classifier.py --train_file data/reviews_ollama_context_labeled_train.jsonl --model_name roberta-base --output_dir models/roberta_fake_ollama_classifier --epoch 10
```

### Training the Explainer

To train the model to generate review classification explanations (determining whether a review is true or false):

```bash
python scripts/train_explainer.py --train_file data/reviews_with_labels_small_train.jsonl --model_name facebook/bart-base --output_dir models/bart_fake_explainer  --epoch 5
```

### Evaluating the Classifier

Evaluate the classifier's performance:

```bash
python scripts/evaluate_classifier.py --model_dir models/roberta_fake_ollama_classifier --test_file data/reviews_ollama_context_labeled_test.jsonl
```

### Evaluating the Explainer

Evaluate the explainer model:

```bash
python scripts/evaluate_explainer.py --model_dir models/bart_fake_explainer --test_file data/reviews_with_labels_small_test.jsonl
```

---

## Project Flow

1. **Download Dataset**:
* Use `download_sephora.py` to download the Sephora review dataset.

2. **Data Preprocessing**:
* Use `prepare_context.py` to load, clean, and process the data. This script merges review and product information into context fields used for training.
* Use `insert_products.py` to insert product information into a vector database for retrieval.

3. **Pseudo-Label Generation**:
* The `generate_pseudo_labels.py` script generates fake/real labels for training based on a specific heuristic.

4. **Dataset Creation**:
* Use `create_small_dataset.py` to create a smaller dataset for quick experimentation and testing (including train/test splits).

5. **Training**:
* You can train a **classifier** (`train_classifier.py`) or an **interpreter** (`train_explainer.py`). The classifier distinguishes between real and fake reviews, while the interpreter generates the reasons behind the classification.

6. **Evaluation**:
* After the model is trained, you can use `evaluate_classifier.py` to evaluate the classifier's performance and `evaluate_explainer.py` to evaluate the interpreter's performance.

7. **Synthetic Fake Review Generation** (Optional):
* Use `generate_synthetic_fakes.py` to use large language models with ollama to generate synthetic fake reviews to augment the training dataset.

---

## Models

This project uses the following models:

- **Classification**: The primary model for review classification is `roberta-base` (for distinguishing real vs fake reviews).
- **Explanation Generation**: For generating explanations behind the classification, we use either `facebook/bart-base` or `t5-base`.
- **Lightweight Classification**: For more efficient and faster classification, you can also use `all-MiniLM-L6-v2`, a smaller model optimized for sentence embeddings and tasks like classification.

---

## Notes

* **Resources**: For large models, you may need GPUs for efficient training.

---

# License

This project is only open to the public; others may use it for research and educational purposes only. Please cite the project if you use it in your work.
