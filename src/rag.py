# src/rag.py  (replacement — paste entire file)
import os
import json
import numpy as np
from typing import Dict, Any, List
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent / "data"
META_PATH = DATA_DIR / "meta.jsonl"
EMB_PATH = DATA_DIR / "embeddings.npy"

# TF-IDF caches
_TFIDF = None
_TFIDF_MATRIX = None
_METAS_CACHE = None

def _load_meta() -> List[Dict[str, Any]]:
    metas = []
    if not META_PATH.exists():
        return metas
    with open(META_PATH, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            metas.append(json.loads(line))
    return metas

def build_tfidf_from_meta(force_rebuild: bool = False):
    """Build and cache TF-IDF vectorizer from meta.jsonl texts."""
    global _TFIDF, _TFIDF_MATRIX, _METAS_CACHE
    if _TFIDF is not None and _TFIDF_MATRIX is not None and not force_rebuild:
        return _TFIDF, _TFIDF_MATRIX, _METAS_CACHE

    metas = _load_meta()
    texts = [m.get("text", "") for m in metas]
    if not texts:
        return None, None, metas

    # improved TF-IDF: include bigrams and reasonable max features
    vect = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=5000)
    mat = vect.fit_transform(texts)  # shape (n_chunks, n_features)

    _TFIDF = vect
    _TFIDF_MATRIX = mat
    _METAS_CACHE = metas
    return _TFIDF, _TFIDF_MATRIX, metas

def embed_query_fallback(query: str):
    """
    Attempt to use SentenceTransformers if available; else use TF-IDF.
    Returns (vector, backend) where backend is 'sbert' or 'tfidf'.
    """
    try:
        from sentence_transformers import SentenceTransformer
        # load on first use; set device cpu to avoid meta-tensor problems
        model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        emb = model.encode(query, convert_to_numpy=True).astype("float32").reshape(1, -1)
        return emb, "sbert"
    except Exception as e:
        # fallback to TF-IDF
        tfidf, mat, metas = build_tfidf_from_meta()
        if tfidf is None:
            raise RuntimeError("No embedding model available and TF-IDF could not be built.")
        qvec = tfidf.transform([query])  # sparse (1, F)
        return qvec, "tfidf"

def retrieve_with_tfidf(query: str, k: int = 5):
    """Retrieve top-k chunks using TF-IDF cosine similarity (robust fallback)."""
    tfidf, mat, metas = build_tfidf_from_meta()
    if tfidf is None or mat is None or len(metas) == 0:
        return []

    qvec = tfidf.transform([query])  # sparse (1, F)
    sims = cosine_similarity(qvec, mat).flatten()  # (n_chunks,)

    # get top k indices sorted by similarity desc
    topk = sims.argsort()[::-1][:k]
    retrieved = []
    seen_texts = set()
    for idx in topk:
        text = metas[idx].get("text", "")
        if text in seen_texts:
            continue
        seen_texts.add(text)
        retrieved.append({
            "page": metas[idx].get("page"),
            "chunk_id": metas[idx].get("chunk_id"),
            "text": text,
            "_score": float(sims[idx])
        })
        if len(retrieved) >= k:
            break
    return retrieved

def retrieve_and_prepare(question: str, k: int = 5, similarity_threshold: float = None) -> Dict[str, Any]:
    """
    Main function used by the app.
    Returns:
      {
        "retrieved": [ {page, chunk_id, text, _score}, ... ],
        "prompt": str,
        "diagnostics": { "backend": "tfidf" or "sbert", "num_chunks": int }
      }
    """
    # Try vectorized/embeddings path if embeddings.npy exists and vector_index is available
    # but keep TF-IDF as guaranteed fallback
    try:
        # prefer using vector_index if embeddings available
        if EMB_PATH.exists():
            # delayed import to avoid heavy modules at import time
            try:
                from vector_index import VectorIndex
                import numpy as np
                embs = np.load(str(EMB_PATH)).astype("float32")
                idx = VectorIndex(embs)
                q_emb, backend = embed_query_fallback(question)
                # if backend == 'tfidf', q_emb is sparse and not usable by vector_index
                if backend == "sbert":
                    ids, dists = idx.search(q_emb, k=k)
                    metas = _load_meta()
                    retrieved = []
                    seen = set()
                    for i, dist in zip(ids, dists):
                        if i < 0 or i >= len(metas):
                            continue
                        text = metas[i].get("text", "")
                        if text in seen: continue
                        seen.add(text)
                        retrieved.append({
                            "page": metas[i].get("page"),
                            "chunk_id": metas[i].get("chunk_id"),
                            "text": text,
                            "_score": float(dist)
                        })
                    # normalize distances -> similarity-like scores (optional)
                    return {
                        "retrieved": retrieved,
                        "prompt": build_prompt(question, retrieved),
                        "diagnostics": {"backend": "vector_index", "num_chunks": len(retrieved)}
                    }
            except Exception as e:
                # any error in vector_index path -> fallback to TF-IDF retrieval
                print("Vector index path failed, falling back to TF-IDF. Error:", e)

        # TF-IDF guaranteed fallback
        retrieved = retrieve_with_tfidf(question, k=k)
        return {
            "retrieved": retrieved,
            "prompt": build_prompt(question, retrieved),
            "diagnostics": {"backend": "tfidf", "num_chunks": len(retrieved)}
        }
    except Exception as e:
        # in case of unexpected error, return empty results with error marker
        print("Error in retrieve_and_prepare:", e)
        return {"retrieved": [], "prompt": build_prompt(question, []), "diagnostics": {"error": str(e)}}

def build_prompt(question: str, retrieved: List[Dict[str, Any]]):
    prompt = "You are an assistant that must answer using ONLY the provided document excerpts.\nIf the answer is not present, say 'I don't know'.\n\n"
    prompt += f"QUESTION: {question}\n\nEVIDENCE:\n"
    for r in retrieved:
        # short snippet + page citation
        snippet = r.get("text", "")
        # optionally truncate long text in prompt to keep it short
        if len(snippet) > 800:
            snippet = snippet[:800] + " ... [truncated]"
        prompt += f"[page {r.get('page')}] {snippet}\n\n"
    prompt += "\nAnswer:"
    return prompt
