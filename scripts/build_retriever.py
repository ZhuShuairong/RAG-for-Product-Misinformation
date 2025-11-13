#!/usr/bin/env python3
"""
build_retriever.py
提供一个简单的检索工具，基于 chromadb 与 sentence-transformers embedding.
Usage:
    from build_retriever import Retriever
    r = Retriever(persist_dir="chroma_db")
    r.retrieve("this product smells like perfume", top_k=3)
"""
import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

class Retriever:
    def __init__(self, persist_dir="chroma_db", model_name="all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_collection("products")

    def retrieve(self, query: str, top_k: int = 3):
        q_emb = self.embedder.encode([query], convert_to_numpy=True)[0].tolist()
        res = self.collection.query(query_embeddings=[q_emb], n_results=top_k, include=["documents", "metadatas"])
        # chroma returns dict with 'documents' as list of lists
        docs = []
        for d, m, id_ in zip(res.get("documents", [[]])[0], res.get("metadatas", [[]])[0], res.get("ids", [[]])[0]):
            docs.append({"id": id_, "document": d, "metadata": m})
        return docs

if __name__ == "__main__":
    r = Retriever()
    q = "This cream smells strongly of perfume and made my skin dry."
    print("Query:", q)
    out = r.retrieve(q, top_k=3)
    for i, o in enumerate(out):
        print("----", i, o["id"])
        print(o["document"][:400])
        print(o["metadata"])
