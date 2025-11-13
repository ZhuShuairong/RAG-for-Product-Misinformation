import chromadb
import pandas as pd
from chromadb.config import Settings
from typing import List, Dict
import os


def initialize_chroma_db(persist_directory: str = "./chroma_db") -> chromadb.Collection:
    """Initialize ChromaDB client and create/get collection"""
    # Create persist directory if it doesn't exist
    os.makedirs(persist_directory, exist_ok=True)
    # Initialize ChromaDB with persistence
    client = chromadb.PersistentClient(path=persist_directory)
    # Create or get collection with cosine similarity
    collection = client.get_or_create_collection(
        name="product_database",
        metadata={"hnsw:space": "cosine"}  # Cosine similarity for embeddings
    )
    print(f"Collection initialized: {collection.name}")
    print(f"Current document count: {collection.count()}")
    return collection


def create_document_text(row: pd.Series) -> str:
    """Create the document text field for embedding"""
    # Handle NaN values
    brand_name = row['brand_name'] if pd.notna(row['brand_name']) else ""
    product_name = row['product_name'] if pd.notna(row['product_name']) else ""
    highlights = row['highlights'] if pd.notna(row['highlights']) else ""
    ingredients = row['ingredients'] if pd.notna(row['ingredients']) else ""
    primary_cat = row['primary_category'] if pd.notna(row['primary_category']) else ""
    secondary_cat = row['secondary_category'] if pd.notna(row['secondary_category']) else ""
    tertiary_cat = row['tertiary_category'] if pd.notna(row['tertiary_category']) else ""
    size = row['size'] if pd.notna(row['size']) else ""
    # Build document text according to specified format
    doc_parts = []
    # Brand and product name
    if brand_name or product_name:
        doc_parts.append(f"{brand_name} {product_name}".strip())
    # Highlights
    if highlights:
        doc_parts.append(f"{highlights}")
    # Ingredients
    if ingredients:
        doc_parts.append(f"Ingredients: {ingredients}")
    # Categories
    categories = " > ".join(filter(None, [primary_cat, secondary_cat, tertiary_cat]))
    if categories:
        doc_parts.append(f"Categories: {categories}")
    # Size
    if size:
        doc_parts.append(f"Size: {size}")
    return ". ".join(doc_parts) + "."


def create_metadata(row: pd.Series) -> Dict:
    """Create metadata dictionary with structured data"""
    metadata = {
        # Core identifiers
        'product_id': str(row['product_id']),
        'brand_name': str(row['brand_name']) if pd.notna(row['brand_name']) else "",

        # Categories for filtering
        'primary_category': str(row['primary_category']) if pd.notna(row['primary_category']) else "",
        'secondary_category': str(row['secondary_category']) if pd.notna(row['secondary_category']) else "",
        'tertiary_category': str(row['tertiary_category']) if pd.notna(row['tertiary_category']) else "",

        # Numerical features (ChromaDB requires specific types)
        'price_usd': float(row['price_usd']) if pd.notna(row['price_usd']) else 0.0,
        'rating': float(row['rating']) if pd.notna(row['rating']) else 0.0,
        'reviews': int(row['reviews']) if pd.notna(row['reviews']) else 0,
        'loves_count': int(row['loves_count']) if pd.notna(row['loves_count']) else 0,

        # Boolean filters
        'limited_edition': bool(row['limited_edition']) if pd.notna(row['limited_edition']) else False,
        'new': bool(row['new']) if pd.notna(row['new']) else False,
        'sephora_exclusive': bool(row['sephora_exclusive']) if pd.notna(row['sephora_exclusive']) else False,

        # Product variations
        'child_count': int(row['child_count']) if pd.notna(row['child_count']) else 0,
        'child_max_price': float(row['child_max_price']) if pd.notna(row['child_max_price']) else 0.0,
        'child_min_price': float(row['child_min_price']) if pd.notna(row['child_min_price']) else 0.0,
    }
    return metadata


def insert_products(collection: chromadb.Collection, csv_path: str, batch_size: int = 100):
    """Insert products from CSV into ChromaDB"""
    # Load CSV
    print(f"Loading products from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} products")
    # Prepare data for insertion
    documents = []
    metadatas = []
    ids = []
    print("Processing products...")
    for idx, row in df.iterrows():
        # Create document text
        doc_text = create_document_text(row)
        documents.append(doc_text)

        # Create metadata
        metadata = create_metadata(row)
        metadatas.append(metadata)

        # Create unique ID
        ids.append(f"prod_{row['product_id']}")

        # Insert in batches
        if len(documents) >= batch_size:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Inserted batch of {len(documents)} products")
            documents, metadatas, ids = [], [], []
    # Insert remaining products
    if documents:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Inserted final batch of {len(documents)} products")
    print(f"\nTotal products in database: {collection.count()}")


def main():
    # Initialize database
    collection = initialize_chroma_db(persist_directory="./chroma_db")
    # Insert products from CSV
    csv_path = "./data/product_info.csv"
    insert_products(collection, csv_path, batch_size=100)
    print("\n✓ Database initialized and populated successfully!")


if __name__ == "__main__":
    main()
