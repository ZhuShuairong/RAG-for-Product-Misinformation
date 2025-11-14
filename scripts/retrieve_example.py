#!/usr/bin/env python3
from build_retriever import Retriever
import argparse

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--query", type=str, required=False, default="This cream smells strongly of perfume and made my skin dry.")
    p.add_argument("--top_k", type=int, default=3)
    args = p.parse_args()
    r = Retriever()
    results = r.retrieve(args.query, top_k=args.top_k)
    for i, rdoc in enumerate(results):
        print(f"### Rank {i + 1} id={rdoc['id']}")
        print(rdoc['document'][:400])
        print("meta:", rdoc['metadata'])
        print()
