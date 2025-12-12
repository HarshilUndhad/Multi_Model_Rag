#Creating embeddings for chunks using sentence-transformers and saving

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

MODEL_NAME = "all-MiniLM-L6-v2" 
#BLOCK 1: read chunks from chunks.jsonl
def read_chunks(chunks_jsonl: str):
    metas = []
    texts = []
    with open(chunks_jsonl, "r", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            metas.append(obj)
            texts.append(obj["text"])
    return texts, metas

#BLOCK 2: embed texts using sentence-transformers
def embed_texts(texts, model_name=MODEL_NAME, batch_size=32):
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, show_progress_bar=True, batch_size=batch_size, convert_to_numpy=True)
    return embs
#BLOCK 3: main function to run the embedding
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", default="data/chunks.jsonl")
    p.add_argument("--out_emb", default="data/embeddings.npy")
    p.add_argument("--out_meta", default="data/meta.jsonl")
    args = p.parse_args()

    texts, metas = read_chunks(args.chunks)
    if not texts:
        raise SystemExit("No chunks found. Run chunker first.")
    embs = embed_texts(texts)

    Path(args.out_emb).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_emb, embs.astype("float32"))
    with open(args.out_meta, "w", encoding="utf-8") as fh:
        for m in metas:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"Saved embeddings ({embs.shape}) to {args.out_emb} and meta to {args.out_meta}")