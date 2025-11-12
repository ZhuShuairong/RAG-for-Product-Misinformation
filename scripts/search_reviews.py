import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import pandas as pd


def load_chroma_db(persist_directory: str = "./chroma_db") -> chromadb.Collection:
    """Load existing ChromaDB collection"""
    
    client = chromadb.PersistentClient(path=persist_directory)
    
    try:
        collection = client.get_collection(name="product_database")
        print(f"Loaded collection: {collection.name}")
        print(f"Total products: {collection.count()}")
        return collection
    except Exception as e:
        raise Exception(f"Failed to load collection: {e}")


def search_products_by_review(
    collection: chromadb.Collection,
    review_text: str,
    k: int = 5,
    filters: Optional[Dict] = None
) -> Dict:
    """
    Search for top k similar products using review text
    
    Args:
        collection: ChromaDB collection
        review_text: The review text to search with
        k: Number of top results to return
        filters: Optional metadata filters (e.g., {"primary_category": "Skincare"})
    
    Returns:
        Dictionary with products, distances, and metadata
    """
    
    # Query the collection
    results = collection.query(
        query_texts=[review_text],
        n_results=k,
        where=filters,  # Optional metadata filtering
        include=['documents', 'metadatas', 'distances']
    )
    
    return results


def format_results(results: Dict, show_distances: bool = True) -> pd.DataFrame:
    """Format search results into a readable DataFrame"""
    
    products = []
    
    for i in range(len(results['ids'][0])):
        product = {
            'rank': i + 1,
            'product_id': results['metadatas'][0][i]['product_id'],
            'brand_name': results['metadatas'][0][i]['brand_name'],
            'primary_category': results['metadatas'][0][i]['primary_category'],
            'secondary_category': results['metadatas'][0][i]['secondary_category'],
            'price_usd': results['metadatas'][0][i]['price_usd'],
            'rating': results['metadatas'][0][i]['rating'],
            'reviews': results['metadatas'][0][i]['reviews'],
            'loves_count': results['metadatas'][0][i]['loves_count'],
        }
        
        if show_distances:
            product['similarity_score'] = 1 - results['distances'][0][i]  # Convert distance to similarity
        
        products.append(product)
    
    return pd.DataFrame(products)


def search_with_filters_example(collection: chromadb.Collection):
    """Example searches with different filters"""
    
    # Example 1: Simple search
    print("\n" + "="*60)
    print("Example 1: Basic Search")
    print("="*60)
    review_text = "I've been using this for a week and my skin is so much softer"
    results = search_products_by_review(collection, review_text, k=5)
    df = format_results(results)
    print(f"\nReview: '{review_text}'")
    print(f"\nTop {len(df)} matching products:")
    print(df.to_string(index=False))
    
    # Example 2: Filter by category
    print("\n" + "="*60)
    print("Example 2: Search with Category Filter")
    print("="*60)
    review_text = "Amazing retinol, no purging, very gentle on skin"
    filters = {"primary_category": "Skincare"}
    results = search_products_by_review(collection, review_text, k=5, filters=filters)
    df = format_results(results)
    print(f"\nReview: '{review_text}'")
    print(f"Filter: Primary Category = Skincare")
    print(f"\nTop {len(df)} matching products:")
    print(df.to_string(index=False))
    
    # Example 3: Filter by price range
    print("\n" + "="*60)
    print("Example 3: Search with Price Filter")
    print("="*60)
    review_text = "Great product for anti-aging, love the price"
    filters = {"$and": [
        {"price_usd": {"$lte": 30.0}},  # Price <= $30
        {"rating": {"$gte": 4.0}}        # Rating >= 4.0
    ]}
    results = search_products_by_review(collection, review_text, k=5, filters=filters)
    df = format_results(results)
    print(f"\nReview: '{review_text}'")
    print(f"Filter: Price <= $30 AND Rating >= 4.0")
    print(f"\nTop {len(df)} matching products:")
    print(df.to_string(index=False))


def batch_search_reviews(
    collection: chromadb.Collection,
    reviews_df: pd.DataFrame,
    review_text_column: str = 'review_text',
    k: int = 5
) -> pd.DataFrame:
    """
    Search for multiple reviews at once
    
    Args:
        collection: ChromaDB collection
        reviews_df: DataFrame containing reviews
        review_text_column: Column name containing review text
        k: Number of top products per review
    
    Returns:
        DataFrame with review and matched products
    """
    
    results_list = []
    
    for idx, row in reviews_df.iterrows():
        review_text = row[review_text_column]
        
        # Search for this review
        results = search_products_by_review(collection, review_text, k=k)
        
        # Get top product
        top_product_id = results['metadatas'][0][0]['product_id']
        top_similarity = 1 - results['distances'][0][0]
        
        results_list.append({
            'review_idx': idx,
            'review_text': review_text[:100] + '...' if len(review_text) > 100 else review_text,
            'matched_product_id': top_product_id,
            'similarity_score': top_similarity,
            'top_k_product_ids': [m['product_id'] for m in results['metadatas'][0]]
        })
    
    return pd.DataFrame(results_list)


def main():
    # Load database
    collection = load_chroma_db(persist_directory="./chroma_db")
    
    # Run example searches
    search_with_filters_example(collection)
    
    # Example batch processing
    print("\n" + "="*60)
    print("Example 4: Batch Review Processing")
    print("="*60)
    
    # Simulate multiple reviews
    sample_reviews = pd.DataFrame({
        'review_text': [
            "Love this retinol serum, very gentle and effective",
            "Horrible smell after reformulation, used to love it",
            "Great for sensitive skin, no irritation at all",
            "Expensive but worth it for anti-aging benefits"
        ]
    })
    
    batch_results = batch_search_reviews(collection, sample_reviews, k=3)
    print("\nBatch processing results:")
    print(batch_results.to_string(index=False))


if __name__ == "__main__":
    main()
