# Phase 3 - RoBERTa Binary Classification + Ollama Explanations (Optimized)
# This phase implements a two-stage pipeline for fake review detection with speed optimizations: 
# Stage 1: RoBERTa Binary Classifier (Optimized) - Uses RoBERTa-base optimized for classification tasks. 
# Stage 2: Ollama (Gemma3:4b) Explanation Generation - Uses Langchain with Ollama for generating customer service-style explanations post-classification.
# Depends on training_examples.json, synthetic_test.csv, and product_info_clean.csv from phase 2.
# Requires Ollama installed and running (ollama serve). The script will automatically check/pull gemma3:4b if needed.

import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from datasets import Dataset, load_from_disk, DatasetDict
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime
import chromadb
import os
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
import subprocess
import sys
import time

# Automated Ollama Setup Function
def setup_ollama_model(model_name="gemma3:4b"):
    """Check if Ollama server is running and model is available; pull if not.
    Note: Assumes 'ollama serve' is run manually or via system service before this script.
    This function checks/pulls the model but does not start the server automatically
    to avoid blocking the script. If server is not running, Ollama calls will fail.
    """
    print(f"Setting up Ollama model: {model_name}")
    
    # Step 1: Check if Ollama server is responsive (simple ping via list command)
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            print("WARNING: Ollama server may not be running. Please start it with 'ollama serve' in a separate terminal.")
            print("Attempting to proceed, but LLM calls will fail if server is down.")
        else:
            print("Ollama server is responsive.")
    except subprocess.TimeoutExpired:
        print("ERROR: Ollama command timed out. Ensure Ollama is installed and server is running.")
        sys.exit(1)
    except FileNotFoundError:
        print("ERROR: 'ollama' command not found. Install Ollama from https://ollama.com/ and ensure it's in PATH.")
        sys.exit(1)
    
    # Step 2: Check if model is listed
    output = result.stdout
    if model_name in output:
        print(f"Model {model_name} is already available.")
        return True
    
    # Step 3: Pull the model if not present
    print(f"Model {model_name} not found. Pulling it now... (This may take a few minutes depending on your connection.)")
    try:
        pull_result = subprocess.run(["ollama", "pull", model_name], capture_output=True, text=True, timeout=300)  # 5-min timeout for pull
        if pull_result.returncode == 0:
            print(f"Successfully pulled {model_name}.")
            # Verify after pull
            time.sleep(2)  # Brief wait for indexing
            verify_result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
            if model_name in verify_result.stdout:
                print(f"Verification: {model_name} is now available.")
                return True
            else:
                print("WARNING: Model pull succeeded, but not listed. Try running again.")
        else:
            print(f"ERROR pulling {model_name}: {pull_result.stderr}")
            sys.exit(1)
    except subprocess.TimeoutExpired:
        print(f"ERROR: Timeout pulling {model_name}. Check your connection or run 'ollama pull {model_name}' manually.")
        sys.exit(1)
    
    return False

# Run Ollama setup
ollama_setup_success = setup_ollama_model("gemma3:4b")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    torch.cuda.empty_cache()
    print("GPU cache cleared")

# Initialize Ollama with Gemma3:4b for explanations (only if setup succeeded)
if ollama_setup_success:
    llm = Ollama(model="gemma3:4b", temperature=0.7)
    print("Loaded Ollama LLM: gemma3:4b")
else:
    print("Ollama setup failed. Explanations will not work. Proceeding with RoBERTa training only.")
    llm = None

# Langchain Prompt Template for customer service-style explanations
explanation_prompt = PromptTemplate(
    input_variables=["product_info", "review_text", "prediction", "rag_context"],
    template="""
You are a formal customer service representative for a beauty product retailer. 
A customer has submitted the following review about the product:

Product Information: {product_info}

Review: {review_text}

Additional Context: {rag_context}

Our classification indicates this review is {prediction} (fake if classified as such, or authentic if real).

Provide a polite, professional response to the customer. Acknowledge their concerns or experience with the product empathetically. Then, if the review is fake, highlight any discrepancies between the review claims and the actual product information (e.g., mismatched ingredients, categories, or unsubstantiated claims) without accusing them of fakeness. If real, validate their feedback and suggest next steps. Keep the response concise (50-150 words), empathetic, and solution-oriented.
"""
)

class RobertaDatasetFormatter:
    """Format inputs for RoBERTa classification.
    Methods
    - format_input(product_info, review_text) -> str
    - tokenize_function(examples) -> dict
    """
    def __init__(self, tokenizer, max_input_length=256):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length

    def format_input(self, product_info, review_text):
        """Return a human-readable input string for the model."""
        input_text = f"Product information: {product_info}\nReview: {review_text}\n\nIs this review authentic for this product?"
        return input_text

    def tokenize_function(self, examples):
        inputs = examples["input_text"]
        labels = examples["label"]
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_input_length,
            truncation=True,
            padding="max_length",
        )
        model_inputs["labels"] = labels
        return model_inputs

def prepare_training_data_roberta(examples, formatter):
    """Prepare lists of input_text and labels from training examples.
    Returns
    - inputs: list[str]
    - labels: list[int]
    """
    inputs = []
    labels = []
    print("Preparing training data for RoBERTa...")
    for i, ex in enumerate(examples):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(examples)} examples")
        input_text = formatter.format_input(ex['product_info'], ex['review_text'])
        inputs.append(input_text)
        labels.append(int(ex['label']))
    return inputs, labels

def compute_metrics(eval_pred):
    """Compute standard classification metrics given (predictions, labels).
    Expects eval_pred: (logits, labels)
    Returns a dict of floats suitable for HF Trainer logging.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(labels, torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy())
    except Exception:
        auc = None
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc) if auc is not None else None,
    }

def _run_model_inference(tokenizer, model, input_texts, device, max_length=256):
    """Tokenize inputs, run the model, and return (pred_labels, confidences, logits).
    - input_texts: list[str]
    - returns: (preds: np.ndarray, confidences: np.ndarray, logits: np.ndarray)
    """
    inputs = tokenizer(input_texts, return_tensors="pt", max_length=max_length, truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.detach().cpu().numpy()
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
        preds = np.argmax(logits, axis=1)
        confidences = probs[np.arange(len(preds)), preds]
    return preds, confidences, logits

def classify_review(product_info, review_text, tokenizer, model, formatter, device):
    """Classify a single review and return ('FAKE'|'REAL', confidence)
    Uses the shared inference helper for consistency and easier testing.
    """
    input_text = formatter.format_input(product_info, review_text)
    preds, confs, _ = _run_model_inference(tokenizer, model, [input_text], device)
    pred = int(preds[0])
    confidence = float(confs[0])
    return ("FAKE" if pred == 1 else "REAL"), confidence

def classify_review_roberta_batch(product_infos, review_texts, tokenizer, model, formatter, device):
    """Batch classify multiple reviews and return (pred_labels, confidences).
    Inputs are parallel lists of product_infos and review_texts.
    """
    input_texts = [formatter.format_input(prod, rev) for prod, rev in zip(product_infos, review_texts)]
    preds, confidences, logits = _run_model_inference(tokenizer, model, input_texts, device)
    return preds.tolist(), confidences.tolist()

def get_rag_context(review_text, product_info=""):
    """Get RAG context for a review by querying ChromaDB"""
    try:
        results = product_profile_collection.query(
            query_texts=[review_text],
            n_results=3
        )
        contexts = results['documents'][0] if results.get('documents') else []
        return " ".join(contexts[:2])
    except Exception as e:
        return f"Error retrieving context: {str(e)}"

def generate_explanation_with_ollama(product_info, review_text, prediction, rag_context):
    """Generate explanation using Ollama/Langchain in customer service style."""
    if llm is None:
        return "Ollama not available. Skipping explanation generation."
    
    pred_text = "fake" if prediction == 1 else "real/authentic"
    try:
        chain = explanation_prompt | llm
        explanation = chain.invoke({
            "product_info": product_info,
            "review_text": review_text,
            "prediction": pred_text,
            "rag_context": rag_context
        })
        return explanation.strip() if explanation.strip() else "Unable to generate explanation."
    except Exception as e:
        return f"Error generating explanation: {str(e)}"

def generate_explanation(review_text, prediction, confidence):
    """Fallback or simple explanation generation using Ollama (without full context)."""
    if llm is None:
        return "Ollama not available. Skipping explanation generation."
    
    # For consistency, use a simplified prompt if no product_info/rag available
    simple_prompt = PromptTemplate(
        input_variables=["review_text", "prediction", "confidence"],
        template="""
You are a formal customer service representative. 
Review: {review_text}
Prediction: {'Real' if prediction == 0 else 'Fake'}
Confidence: {confidence:.3f}

Provide a polite response acknowledging the customer's feedback. If fake, note any potential discrepancies subtly. Keep concise and empathetic.
"""
    )
    try:
        chain = simple_prompt | llm
        explanation = chain.invoke({
            "review_text": review_text,
            "prediction": prediction,
            "confidence": confidence
        })
        return explanation.strip()
    except Exception as e:
        return f"Error generating explanation: {str(e)}"

def explain_prediction(review_text, prediction, confidence):
    try:
        explanation = generate_explanation(review_text, prediction, confidence)
        return explanation if explanation else "Unable to generate explanation."
    except Exception as e:
        return f"Error generating explanation: {str(e)}"

def setup_batch_evaluation(test_df, batch_size=32):
    predictions = []
    ground_truth = []
    explanations = []
    confidences = []
    print(f"Setting up batch evaluation with batch size {batch_size}")
    print(f"Total test samples: {len(test_df)}")
    return predictions, ground_truth, explanations, confidences, batch_size

def get_rag_contexts_for_fake_reviews(fake_reviews):
    batch_rag_contexts = []
    for review_text in fake_reviews:
        rag_context = get_rag_context(review_text, "")
        batch_rag_contexts.append(rag_context)
    return batch_rag_contexts

def generate_explanations_for_fake_reviews(batch_df, fake_indices, batch_rag_contexts):
    fake_explanations = []
    for i, idx in enumerate(fake_indices):
        row = batch_df.iloc[idx]
        explanation = generate_explanation_with_ollama(
            row['product_info'],
            row['review_text'],
            1,
            batch_rag_contexts[i],
        )
        fake_explanations.append(explanation)
    return fake_explanations

def combine_batch_explanations(batch_df, batch_predictions, fake_indices, fake_explanations):
    batch_explanations = []
    explanation_idx = 0
    for i in range(len(batch_df)):
        if i in fake_indices:
            batch_explanations.append(fake_explanations[explanation_idx])
            explanation_idx += 1
        else:
            # For real reviews, generate a validating response
            row = batch_df.iloc[i]
            explanation = generate_explanation_with_ollama(
                row['product_info'],
                row['review_text'],
                0,
                get_rag_context(row['review_text'], row['product_info'])
            )
            batch_explanations.append(explanation)
    return batch_explanations

def evaluate_model(test_df, model, tokenizer, chromadb_client, formatter, batch_size=32):
    if llm is None:
        print("Ollama not available. Running classification only (no explanations).")
        predictions = []
        ground_truth = []
        confidences = []
        print("Starting batch evaluation (classification only)...")
        for start_idx in tqdm(range(0, len(test_df), batch_size), desc="Processing batches"):
            end_idx = min(start_idx + batch_size, len(test_df))
            batch_df = test_df.iloc[start_idx:end_idx]
            batch_product_infos = batch_df['product_info'].tolist()
            batch_review_texts = batch_df['review_text'].tolist()
            batch_preds, batch_confs = classify_review_roberta_batch(batch_product_infos, batch_review_texts, tokenizer, model, formatter, device)
            predictions.extend(batch_preds)
            confidences.extend(batch_confs)
            ground_truth.extend(batch_df['is_fake'].astype(int).tolist())
        explanations = ["Explanations disabled due to Ollama unavailability."] * len(predictions)
        return predictions, ground_truth, explanations, confidences
    
    predictions, ground_truth, explanations, confidences, _ = setup_batch_evaluation(test_df, batch_size)
    print("Starting batch evaluation...")
    for start_idx in tqdm(range(0, len(test_df), batch_size), desc="Processing batches"):
        end_idx = min(start_idx + batch_size, len(test_df))
        batch_df = test_df.iloc[start_idx:end_idx]
        batch_product_infos = batch_df['product_info'].tolist()
        batch_review_texts = batch_df['review_text'].tolist()
        batch_preds, batch_confs = classify_review_roberta_batch(batch_product_infos, batch_review_texts, tokenizer, model, formatter, device)
        fake_indices = [i for i, pred in enumerate(batch_preds) if pred == 1]
        if fake_indices:
            fake_reviews = batch_df.iloc[fake_indices]['review_text'].tolist()
            batch_rag_contexts = get_rag_contexts_for_fake_reviews(fake_reviews)
            fake_explanations = generate_explanations_for_fake_reviews(batch_df, fake_indices, batch_rag_contexts)
            batch_explanations = combine_batch_explanations(batch_df, batch_preds, fake_indices, fake_explanations)
        else:
            # If no fakes, generate explanations for all (reals)
            batch_explanations = []
            for _, row in batch_df.iterrows():
                explanation = generate_explanation_with_ollama(
                    row['product_info'],
                    row['review_text'],
                    0,
                    get_rag_context(row['review_text'], row['product_info'])
                )
                batch_explanations.append(explanation)
        predictions.extend(batch_preds)
        confidences.extend(batch_confs)
        explanations.extend(batch_explanations)
        ground_truth.extend(batch_df['is_fake'].astype(int).tolist())
    return predictions, ground_truth, explanations, confidences

# Data loading and preparation
with open('./data/training_examples.json', 'r') as f:
    training_examples = json.load(f)

print(f"Loaded {len(training_examples)} training examples")

# Load test_df from Phase 2 CSV
test_df = pd.read_csv('./data/synthetic_test.csv')
print(f"Loaded {len(test_df)} test samples from synthetic_test.csv")

# Load product_info_clean to merge for test_df
product_info_clean = pd.read_csv('./data/product_info_clean.csv')

# Merge test_df with product_info to add product_info column (matching Phase 2 format)
test_df = test_df.merge(
    product_info_clean[['product_id', 'product_name', 'brand', 'primary_category', 'price', 'ingredients', 'highlights']],
    on='product_id', how='left'
)

# Create 'product_info' column for test_df
def create_product_info(row):
    return f"Product Name: {row.get('product_name', 'N/A')}\nBrand: {row.get('brand', 'N/A')}\nCategory: {row.get('primary_category', 'N/A')}\nPrice: {row.get('price', 'N/A')}\nIngredients: {row.get('ingredients', 'N/A')}\nHighlights: {row.get('highlights', 'N/A')}"

test_df['product_info'] = test_df.apply(create_product_info, axis=1)
test_df = test_df.drop(columns=['product_name', 'brand', 'primary_category', 'price', 'ingredients', 'highlights'])  # Clean up temp columns
print("Merged test_df with product info.")

client = chromadb.PersistentClient(path="./data/chroma_data")
product_profile_collection = client.get_collection(name="product_profiles")

# Debug: Check training data labels
print("Sample training examples:")
for i, ex in enumerate(training_examples[:5]):
    print(f"Example {i}: Label={ex['label']}, Product={ex['product_info'][:100]}..., Review={ex['review_text'][:100]}...")

# Check label distribution
labels = [ex['label'] for ex in training_examples]
print(f"\nLabel distribution: {np.bincount(labels)} (0=real, 1=fake)")
print(f"Label ratio: {np.mean(labels):.3f} fake")

# DEBUG: Create a small balanced debug dataset if possible
print("\n" + "="*50)
print("DEBUG: Creating small balanced dataset")
print("="*50)

# Sample small balanced dataset
debug_size = 1000  # Target size
real_examples = [ex for ex in training_examples if ex['label'] == 0]
fake_examples = [ex for ex in training_examples if ex['label'] == 1]

# Determine feasible balanced size
min_class = min(len(real_examples), len(fake_examples))
if min_class == 0:
    print("Not enough examples to create a balanced debug dataset; skipping debug sampling.")
else:
    actual_half = min(min_class, debug_size // 2)
    np.random.seed(42)
    debug_real = np.random.choice(real_examples, size=actual_half, replace=False).tolist()
    debug_fake = np.random.choice(fake_examples, size=actual_half, replace=False).tolist()

    debug_training_examples = debug_real + debug_fake
    np.random.shuffle(debug_training_examples)

    print(f"Debug dataset: {len(debug_training_examples)} examples ({actual_half} real, {actual_half} fake)")

    # COMMENTED OUT: Override training_examples for debugging
    # training_examples = debug_training_examples

print("Using FULL training dataset for production training...")

# Debug: Check what the formatted inputs look like
print("\nSample formatted inputs:")

# Ensure `tokenizer` is available. If it's missing, try to load a saved tokenizer from
# the local `./fake_review_detector_roberta` directory (present in the repo),
# otherwise fall back to the public 'roberta-base' tokenizer.
try:
    tokenizer
except NameError:
    try:
        from transformers import RobertaTokenizer
    except Exception as e:
        raise ImportError("transformers not available: " + str(e))
    import os
    model_dir = os.path.join('.', 'fake_review_detector_roberta')
    if os.path.isdir(model_dir):
        tokenizer = RobertaTokenizer.from_pretrained(model_dir)
        print(f"Loaded tokenizer from {model_dir}")
    else:
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        print("Loaded tokenizer 'roberta-base' as fallback")

# Instantiate the formatter (assumes the class RobertaDatasetFormatter is defined earlier in the notebook)
try:
    formatter = RobertaDatasetFormatter(tokenizer)
except NameError:
    raise NameError("RobertaDatasetFormatter is not defined. Please run the cell that defines this class before running this debug cell.")

# Show a few formatted examples and tokenization result
for i, ex in enumerate(training_examples[:3]):
    product_info = ex.get('product_info', '')
    review_text = ex.get('review_text', '')
    input_text = formatter.format_input(product_info, review_text)
    print(f"--- Example {i} ---")
    print(input_text)
    tokens = tokenizer.tokenize(input_text)
    print("Tokens:", tokens)

# Model and tokenizer setup - Using RoBERTa-base for classification
model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()

# Move model to device
model = model.to(device)
print(f"Model moved to device: {next(model.parameters()).device}")

formatter = RobertaDatasetFormatter(tokenizer)

print(f"\nLoaded model: {model_name}")
print(f"Model device: {next(model.parameters()).device}")

# Check if tokenized dataset already exists
force_regenerate = True

tokenized_dataset_path = "./data/tokenized_roberta_dataset"
if os.path.exists(tokenized_dataset_path) and not force_regenerate:
    print("Loading pre-tokenized dataset...")
    tokenized_datasets = load_from_disk(tokenized_dataset_path)
    train_dataset = tokenized_datasets["train"]
    val_dataset = tokenized_datasets["validation"]
else:
    # Prepare training data
    train_inputs, train_labels = prepare_training_data_roberta(training_examples, formatter)

    train_dataset = Dataset.from_dict({
        "input_text": train_inputs,
        "label": train_labels
    })

    print(f"Training dataset size: {len(train_dataset)}")

    # Train/validation split
    print("\n" + "="*70)
    print("Creating Train/Validation Split")
    print("="*70)

    # Determine whether stratification is feasible
    try:
        label_counts = np.bincount(train_labels)
        use_stratify = label_counts.min() >= 2
    except Exception:
        use_stratify = False

    if use_stratify:
        stratify_labels = train_labels
        print("Using stratified split for tokenization precompute")
    else:
        stratify_labels = None
        print("Not enough examples for stratified split; using random split")

    train_split_indices, val_split_indices = train_test_split(
        range(len(train_dataset)),
        test_size=0.2,
        stratify=stratify_labels,
        random_state=42
    )

    train_split = train_dataset.select(train_split_indices)
    val_split = train_dataset.select(val_split_indices)

    print(f"Training split size: {len(train_split)}")
    print(f"Validation split size: {len(val_split)}")

    # Tokenize datasets (precompute for speed)
    print("\nTokenizing datasets (this may take a moment)...")
    tokenized_train_dataset = train_split.map(formatter.tokenize_function, batched=True, num_proc=1)
    tokenized_val_dataset = val_split.map(formatter.tokenize_function, batched=True, num_proc=1)

    # Save tokenized datasets
    tokenized_datasets = DatasetDict({
        "train": tokenized_train_dataset,
        "validation": tokenized_val_dataset
    })
    tokenized_datasets.save_to_disk(tokenized_dataset_path)
    print(f"Tokenized datasets saved to {tokenized_dataset_path}")

    train_dataset = tokenized_train_dataset
    val_dataset = tokenized_val_dataset

# Training arguments and trainer setup (optimized for speed)
from sklearn.utils.class_weight import compute_class_weight

# Determine batch sizes depending on device
if device.type == "cpu":
    train_batch_size = 4
    eval_batch_size = 4
    grad_accum_steps = 4
else:
    train_batch_size = 64  # Increased from 16
    eval_batch_size = 64   # Increased from 16
    grad_accum_steps = 1   # Reduced from 2

# Compute class weights from the raw training labels (train_labels is defined earlier when preparing inputs)
try:
    unique_classes = np.unique(train_labels)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=np.array(train_labels))
    class_weights_dict = {int(c): float(w) for c, w in zip([0, 1], class_weights)}
    print(f"Computed class weights: {class_weights_dict}")
except Exception as e:
    print(f"Warning: could not compute class weights (falling back to 1.0): {e}")
    class_weights = np.array([1.0, 1.0])

# Move class weights to device for loss calculation
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

# Use weighted loss via a custom Trainer to avoid modifying model code
from transformers import Trainer

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Custom compute_loss that accepts extra kwargs from HF Trainer.
        This avoids TypeError when the Trainer passes framework-specific keywords
        such as `num_items_in_batch`.
        """
        # Trainer already moves tensors to device; labels are in inputs['labels']
        labels = inputs.get("labels")

        # Forward pass
        outputs = model(**inputs)
        # If model returned a precomputed loss (some models do), prefer that when labels are missing
        logits = getattr(outputs, "logits", None)

        if labels is None:
            # If model provided a loss (e.g., when labels are embedded), use it; otherwise fallback to 0
            loss = getattr(outputs, "loss", None)
            if loss is None:
                # Fallback: zero tensor on same device as model
                loss = torch.tensor(0.0, device=next(model.parameters()).device)
        else:
            # Use CrossEntropyLoss with class weights for multi-class/binary classification
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
            # Ensure logits exist
            if logits is None:
                outputs = model(**inputs)
                logits = outputs.logits
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

# TrainingArguments - prefer accuracy or a composite metric for early stopping when F1 may be zero early
training_args = TrainingArguments(
    output_dir="./data/fake_review_detector_roberta",
    num_train_epochs=3,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    gradient_accumulation_steps=grad_accum_steps,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=1,  # Changed from 50 to 1 to see training loss
    warmup_steps=200,
    weight_decay=0.01,
    logging_dir="./logs",
    seed=42,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",  # Use accuracy (or 'f1' once F1 becomes meaningful)
    greater_is_better=True,
    # Speed optimizations
    fp16=True,  # Mixed precision training
    dataloader_num_workers=4,  # Parallel data loading
    dataloader_pin_memory=True,  # Faster GPU transfer
)

# Instantiate our WeightedTrainer instead of the default Trainer
# Use a plain EarlyStoppingCallback() without keyword args to avoid mismatches across transformer versions
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback()]
)

print("Trainer initialized with weighted loss and metric_for_best_model=\"accuracy\"")

# Debug: Check dataset sizes
print("Dataset sizes:")
print(f"Train dataset: {len(train_dataset)} samples")
print(f"Validation dataset: {len(val_dataset)} samples")
print(f"Test dataframe: {len(test_df)} samples")

# Check label distributions
train_labels = [ex['label'] for ex in train_dataset]
val_labels = [ex['label'] for ex in val_dataset]
test_labels = test_df['is_fake'].astype(int).tolist()

print(f"\nTrain labels distribution: {np.bincount(train_labels)} (0=real, 1=fake)")
print(f"Val labels distribution: {np.bincount(val_labels)} (0=real, 1=fake)")
print(f"Test labels distribution: {np.bincount(test_labels)} (0=real, 1=fake)")

# Check data quality: duplicates and label leakage
print("\nData Quality Checks:")

# Check for duplicates in training data (using original training_examples)
train_texts = [ex['review_text'] for ex in training_examples[:len(train_dataset)]]  # Match the split size
val_texts = [ex['review_text'] for ex in training_examples[len(train_dataset):len(train_dataset)+len(val_dataset)]]  # Val split
test_texts = test_df['review_text'].tolist()

print(f"Train duplicates: {len(train_texts) - len(set(train_texts))}")
print(f"Val duplicates: {len(val_texts) - len(set(val_texts))}")
print(f"Test duplicates: {len(test_texts) - len(set(test_texts))}")

# Check for overlap between splits
train_set = set(train_texts)
val_set = set(val_texts)
test_set = set(test_texts)

train_val_overlap = len(train_set.intersection(val_set))
train_test_overlap = len(train_set.intersection(test_set))
val_test_overlap = len(val_set.intersection(test_set))

print(f"Train-Val overlap: {train_val_overlap}")
print(f"Train-Test overlap: {train_test_overlap}")
print(f"Val-Test overlap: {val_test_overlap}")

# Check label balance
print(f"\nLabel balance check:")
print(f"Train fake ratio: {np.mean([ex['label'] for ex in train_dataset]):.3f}")
print(f"Val fake ratio: {np.mean([ex['label'] for ex in val_dataset]):.3f}")
print(f"Test fake ratio: {test_df['is_fake'].mean():.3f}")

# Start training
import traceback
print("\nStarting RoBERTa training (optimized)...")
try:
    train_result = trainer.train()
    best_metric = getattr(trainer.state, 'best_metric', None)
    if best_metric is not None:
        try:
            print(f"Training completed! Best metric: {best_metric:.4f}")
        except Exception:
            print("Training completed! Best metric:", best_metric)
    else:
        print("Training completed! No best_metric available in trainer.state.")
except Exception as e:
    print("Training failed with exception:")
    traceback.print_exc()
    # Re-raise to surface the error to the notebook if desired
    raise

# Save model
trainer.save_model("./data/fake_review_detector_roberta")
tokenizer.save_pretrained("./data/fake_review_detector_roberta")
print("RoBERTa model saved successfully to './data/fake_review_detector_roberta'")

# Optional: Run evaluation on test set (uncomment to execute)
# print("\nRunning evaluation on test set...")
# predictions, ground_truth, explanations, confidences = evaluate_model(
#     test_df, model, tokenizer, client, formatter, batch_size=32
# )
# print(f"Test Accuracy: {accuracy_score(ground_truth, predictions):.4f}")
# print(f"Test F1: {f1_score(ground_truth, predictions, zero_division=0):.4f}")
# print("Evaluation complete. Sample predictions and explanations saved in variables.")

# This phase creates files: fake_review_detector_roberta (model directory) and tokenized_roberta_dataset in the data folder.
