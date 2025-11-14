import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


class Retriever:
    def __init__(self, persist_dir="chroma_db", model_name="all-MiniLM-L6-v2", model_dir=None):
        self.embedder = SentenceTransformer(model_dir if model_dir else model_name)
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
        # Generate the query embedding using the sentence transformer
        q_emb = self.embedder.encode([query], convert_to_numpy=True)[0].tolist()

        # Query ChromaDB for the top_k most relevant documents and their metadata
        res = self.collection.query(query_embeddings=[q_emb], n_results=top_k, include=["documents", "metadatas"])

        # Process the response to extract the documents and their corresponding metadata
        docs = []
        for d, m, id_ in zip(res.get("documents", [[]])[0], res.get("metadatas", [[]])[0], res.get("ids", [[]])[0]):
            docs.append({
                "id": id_,
                "document": d,
                "metadata": m  # The metadata contains all product-related information
            })
        return docs


if __name__ == "__main__":
    r = Retriever(model_dir="./models/all-MiniLM-L6-v2")
    q = "This cream smells strongly of perfume and made my skin dry."
    print("Query:", q)
    out = r.retrieve(q, top_k=3)
    for i, o in enumerate(out):
        print("----", i, o["id"])
        print(o["document"])
        print(o["metadata"])  # Print the product metadata for each document
