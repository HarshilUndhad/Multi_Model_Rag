# src/rag.py
import os
import json
import numpy as np
from typing import Dict, Any, List
from pathlib import Path

# vector_index import (the fallback we've added earlier)
from vector_index import VectorIndex  # optional, used if embeddings.npy is present

# TF-IDF fallback imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# SentenceTransformers attempt
_EMBED_MODEL = None
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent / "data"
EMB_PATH = DATA_DIR / "embeddings.npy"
META_PATH = DATA_DIR / "meta.jsonl"

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

# Try to load SentenceTransformers model (CPU). If it fails, we'll use TF-IDF fallback.
def get_sentence_transformer():
    global _EMBED_MODEL
    if _EMBED_MODEL is not None:
        return _EMBED_MODEL
    try:
        from sentence_transformers import SentenceTransformer
        # force CPU device to reduce likelihood of meta-tensor device ops:
        _EMBED_MODEL = SentenceTransformer(_EMBED_MODEL_NAME, device="cpu")
        return _EMBED_MODEL
    except Exception as e:
        # fail silently and allow TF-IDF fallback
        print("Warning: SentenceTransformer failed to load, falling back to TF-IDF. Error:", e)
        _EMBED_MODEL = None
        return None

# Build TF-IDF vectorizer from meta chunks (cached)
_TFIDF = None
_TFIDF_MATRIX = None
_METAS_CACHE = None

def build_tfidf_from_meta():
    global _TFIDF, _TFIDF_MATRIX, _METAS_CACHE
    if _TFIDF is not None and _TFIDF_MATRIX is not None:
        return _TFIDF, _TFIDF_MATRIX, _METAS_CACHE
    metas = _load_meta()
    texts = [m.get("text", "") for m in metas]
    if not texts:
        return None, None, metas
    vect = TfidfVectorizer(stop_words="english", max_features=20000)
    mat = vect.fit_transform(texts)  # shape (n_chunks, n_features)
    _TFIDF = vect
    _TFIDF_MATRIX = mat
    _METAS_CACHE = metas
    return _TFIDF, _TFIDF_MATRIX, metas

def embed_query_with_fallback(query: str):
    """
    Returns a (1, D) numpy array embedding for the query. 
    Tries SentenceTransformer; if fails, uses TF-IDF vector (sparse -> dense).
    """
    model = get_sentence_transformer()
    if model is not None:
        # try to encode with SentenceTransformer
        try:
            emb = model.encode(query, convert_to_numpy=True)
            if emb is not None:
                return np.asarray(emb, dtype="float32"), "sbert"
        except Exception as e:
            print("Warning: SentenceTransformer encoding failed, falling back to TF-IDF. Error:", e)
            # fall through to TF-IDF fallback

    # TF-IDF fallback
    tfidf, mat, metas = build_tfidf_from_meta()
    if tfidf is None or mat is None:
        raise RuntimeError("No embedding model available and TF-IDF could not be built (meta missing).")
    qvec = tfidf.transform([query])  # sparse matrix (1, features)
    # convert to dense numpy array
    qdense = qvec.toarray().astype("float32")
    return qdense, "tfidf"

def retrieve_with_fallback(query: str, k: int = 5, similarity_threshold: float = None):
    """
    Main retrieval worker. Tries to use precomputed embeddings + VectorIndex if available,
    else uses embed_query_with_fallback + cosine similarity against meta texts (TF-IDF path).
    Returns list of retrieved dicts with keys: page, chunk_id, text, _score
    """
    # If embeddings.npy and VectorIndex is present, prefer vector-based search with VectorIndex
    if EMB_PATH.exists() and META_PATH.exists():
        try:
            # try to load vector index (this will use faiss if available or sklearn fallback)
            import numpy as np
            embs = np.load(EMB_PATH)
            idx = VectorIndex(embs.astype("float32"))
            # use sbert if available for embedding, else tfidf
            q_emb, backend = embed_query_with_fallback(query)
            ids, dists = idx.search(q_emb, k=k)
            # normalize distances -> score (smaller distance better). For FAISS L2 lower is better; for sklearn we returned distances
            retrieved = []
            metas = _load_meta()
            for i, dist in zip(ids, dists):
                if i < 0 or i >= len(metas):
                    continue
                retrieved.append({
                    "page": metas[i].get("page"),
                    "chunk_id": metas[i].get("chunk_id"),
                    "text": metas[i].get("text"),
                    "_score": float(dist)
                })
            return retrieved
        except Exception as e:
            # fallback to TF-IDF path if any error
            print("Warning: VectorIndex path failed, falling back to TF-IDF retrieval. Error:", e)

    # TF-IDF retrieval
    qvec, backend = embed_query_with_fallback(query)  # will return dense (1, D) for tfidf or sbert
    if backend == "tfidf":
        # use cosine similarity between qvec (dense) and TF-IDF matrix (sparse)
        tfidf, mat, metas = build_tfidf_from_meta()
        # mat is sparse (n_chunks, features)
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(qvec, mat)  # shape (1, n_chunks)
        sims = sims.flatten()
        # get top k indices by similarity
        topk_idx = sims.argsort()[::-1][:k]
        retrieved = []
        for idx in topk_idx:
            retrieved.append({
                "page": metas[idx].get("page"),
                "chunk_id": metas[idx].get("chunk_id"),
                "text": metas[idx].get("text"),
                "_score": float(sims[idx])
            })
        return retrieved
    else:
        # if backend == sbert but we couldn't use VectorIndex, fall back to scoring against meta texts by encoding all meta texts
        metas = _load_meta()
        texts = [m.get("text", "") for m in metas]
        try:
            # encode texts (may be heavy, but meta is limited in size)
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(_EMBED_MODEL_NAME, device="cpu")
            txt_embs = model.encode(texts, convert_to_numpy=True)
            from sklearn.metrics.pairwise import cosine_similarity
            sims = cosine_similarity(qvec, txt_embs).flatten()
            topk_idx = sims.argsort()[::-1][:k]
            retrieved = []
            for idx in topk_idx:
                retrieved.append({
                    "page": metas[idx].get("page"),
                    "chunk_id": metas[idx].get("chunk_id"),
                    "text": metas[idx].get("text"),
                    "_score": float(sims[idx])
                })
            return retrieved
        except Exception as e:
            # last-resort: simple substring match ranking
            retrieved = []
            for m in metas[:k]:
                retrieved.append({"page": m.get("page"), "chunk_id": m.get("chunk_id"), "text": m.get("text"), "_score": 0.0})
            return retrieved

def build_prompt_from_chunks(question: str, retrieved: List[Dict[str, Any]]):
    prompt = "You are an assistant that must answer using ONLY the provided document excerpts.\nIf the answer is not present, say 'I don't know'.\n\n"
    prompt += f"QUESTION: {question}\n\nEVIDENCE:\n"
    for r in retrieved:
        prompt += f"[page {r.get('page')}] {r.get('text')}\n\n"
    prompt += "\nAnswer:"
    return prompt

def retrieve_and_prepare(question: str, k: int = 5, similarity_threshold: float = None):
    retrieved = retrieve_with_fallback(question, k=k, similarity_threshold=similarity_threshold)
    prompt = build_prompt_from_chunks(question, retrieved)
    return {"retrieved": retrieved, "prompt": prompt}
