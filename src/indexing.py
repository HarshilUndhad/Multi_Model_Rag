#building and querrying FAISS index


import faiss
import numpy as np
from pathlib import Path
import json

#BLOCK 1: build FAISS index from embeddings and save to disk
def build_index(emb_path: str, meta_path: str, index_path: str = "data/faiss.index"):
    embs = np.load(emb_path).astype("float32")
    # Normalize for cosine similarity with inner product index
    faiss.normalize_L2(embs)
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embs)
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)
    print(f"Built FAISS index with {index.ntotal} vectors and saved to {index_path}")

#BLOCK 2: load FAISS index and metadata
def load_index(index_path: str = "data/faiss.index", meta_path: str = "data/meta.jsonl"):
    index = faiss.read_index(index_path)
    metas = []
    with open(meta_path, "r", encoding="utf-8") as fh:
        for line in fh:
            metas.append(json.loads(line))
    return index, metas

#BLOCK 3: search FAISS index with a query vector
def search_index(query_vec, k=5, index_path="data/faiss.index", meta_path="data/meta.jsonl"):
    index, metas = load_index(index_path, meta_path)
    q = np.array([query_vec]).astype("float32")
    faiss.normalize_L2(q)
    D, I = index.search(q, k)
    results = []
    for idx in I[0]:
        results.append(metas[idx])
    return results

#BLOCK 4: main function to build index from embeddings and metadata
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--emb", default="data/embeddings.npy")
    p.add_argument("--index", default="data/faiss.index")
    p.add_argument("--meta", default="data/meta.jsonl")
    args = p.parse_args()
    build_index(args.emb, args.meta, args.index)
