# src/rag.py
"""
Robust retrieval + rerank + strict prompt builder for RAG pipeline.
Returns: dict with keys:
  - retrieved: list[ {page, chunk_id, text, _score, _rerank_score?} ]
  - prompt: str (strict evidence-only prompt)
  - diagnostics: { backend: 'vector'|'sbert'|'tfidf', num_chunks: int, notes: ... }
"""

from pathlib import Path
import json
import os
from typing import List, Dict, Any, Tuple

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent / "data"
META_PATH = DATA_DIR / "meta.jsonl"
EMB_PATH = DATA_DIR / "embeddings.npy"

# Delayed imports to avoid heavy imports at module import time
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
    global _TFIDF, _TFIDF_MATRIX, _METAS_CACHE
    if _TFIDF is not None and _TFIDF_MATRIX is not None and not force_rebuild:
        return _TFIDF, _TFIDF_MATRIX, _METAS_CACHE

    metas = _load_meta()
    texts = [m.get("text", "") for m in metas]
    if not texts:
        return None, None, metas

    from sklearn.feature_extraction.text import TfidfVectorizer
    vect = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=5000)
    mat = vect.fit_transform(texts)
    _TFIDF = vect
    _TFIDF_MATRIX = mat
    _METAS_CACHE = metas
    return _TFIDF, _TFIDF_MATRIX, metas

def embed_query_try_sbert(query: str) -> Tuple[Any, str]:
    """
    Try to get an embedding using SentenceTransformer on CPU.
    Returns (embedding, backend) where embedding is numpy array (1, D) or None.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        emb = model.encode(query, convert_to_numpy=True).reshape(1, -1)
        return emb, "sbert"
    except Exception as e:
        # sbert failed — fall back to TF-IDF
        return None, "sbert_failed"

def retrieve_with_vector_index(query_emb, k=5):
    """
    Use src/vector_index.VectorIndex if available and embeddings.npy exists.
    Returns retrieved list or raises.
    """
    if not EMB_PATH.exists():
        raise FileNotFoundError("embeddings.npy not found")
    try:
        import numpy as np
        embs = np.load(str(EMB_PATH)).astype("float32")
        # dynamic import of vector_index
        from vector_index import VectorIndex
        idx = VectorIndex(embs)
        ids, dists = idx.search(query_emb, k=k)
        metas = _load_meta()
        retrieved = []
        seen = set()
        for i, d in zip(ids, dists):
            if i < 0 or i >= len(metas):
                continue
            text = metas[i].get("text","")
            if text in seen:
                continue
            seen.add(text)
            retrieved.append({"page": metas[i].get("page"), "chunk_id": metas[i].get("chunk_id"),
                              "text": text, "_score": float(d)})
        return retrieved
    except Exception as e:
        raise

def retrieve_with_tfidf_scoring(query: str, k=10):
    """
    Return top-k candidates using TF-IDF cosine similarity.
    """
    tfidf, mat, metas = build_tfidf_from_meta()
    if tfidf is None:
        return []
    from sklearn.metrics.pairwise import cosine_similarity
    qvec = tfidf.transform([query])
    sims = cosine_similarity(qvec, mat).flatten()
    topk = sims.argsort()[::-1][:k]
    retrieved = []
    seen = set()
    for idx in topk:
        text = metas[idx].get("text","")
        if text in seen: continue
        seen.add(text)
        retrieved.append({"page": metas[idx].get("page"), "chunk_id": metas[idx].get("chunk_id"),
                          "text": text, "_score": float(sims[idx])})
        if len(retrieved) >= k:
            break
    return retrieved

def re_rank_by_tfidf(query: str, candidates: List[Dict[str, Any]], top_k: int = 5):
    """
    Re-rank the candidate snippets by TF-IDF similarity to query (cross-encoder-style re-rank).
    Returns top_k candidates with added '_rerank_score'.
    """
    if not candidates:
        return []
    texts = [c.get("text","") for c in candidates]
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vect = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=5000)
    all_docs = [query] + texts
    mat = vect.fit_transform(all_docs)
    qvec = mat[0]
    docs_mat = mat[1:]
    sims = cosine_similarity(qvec, docs_mat).flatten()
    idxs = sims.argsort()[::-1][:min(top_k, len(texts))]
    reranked = []
    for i in idxs:
        c = candidates[i]
        c = c.copy()
        c["_rerank_score"] = float(sims[i])
        reranked.append(c)
    return reranked

def build_strict_prompt(question: str, retrieved: List[Dict[str, Any]], max_chars_per_chunk: int = 800):
    header = (
        "You are a fact-checking assistant. Answer the QUESTION using ONLY the EVIDENCE blocks below.\n"
        "Do NOT invent facts. If the evidence does not contain an answer, reply exactly: I don't know.\n"
        "Cite evidence inline using [page X — chunk Y]. Provide:\n"
        "1) One-line concise answer (if available).\n"
        "2) Quoted evidence lines (copy-paste from evidence) that support the answer.\n"
        "3) Short attribution line with sources used.\n\n"
    )
    q = f"QUESTION: {question}\n\nEVIDENCE:\n"
    for r in retrieved:
        text = r.get("text","")
        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk] + " ... [truncated]"
        q += f"[page {r.get('page')} — {r.get('chunk_id')}] {text}\n\n"
    q += "\nAnswer now.\n"
    return header + q

def verify_answer_against_evidence(answer: str, retrieved: List[Dict[str, Any]]) -> bool:
    """
    Heuristic verification:
      - If numeric tokens present in answer, ensure they appear in any retrieved text.
      - For named entity-like tokens, require some overlap.
    Returns True if passes heuristic, False otherwise.
    """
    import re
    if not retrieved:
        return False
    combined = " ".join([r.get("text","").lower() for r in retrieved])
    # numbers
    numbers = re.findall(r"\d+\.?\d*", answer)
    if numbers:
        for n in numbers:
            if n in combined:
                return True
        return False
    # named tokens
    tokens = [t.lower() for t in answer.split() if len(t) > 3]
    if not tokens:
        return False
    overlap = sum(1 for t in tokens if t in combined)
    return overlap >= max(1, len(tokens)//6)

def retrieve_and_prepare(question: str, k: int = 5, use_vector_index_first: bool = True) -> Dict[str, Any]:
    """
    Main entry for app. Returns retrieved snippets, prompt, and diagnostics.
    """
    diagnostics = {"backend": None, "notes": []}
    # First try vector path if available
    candidates = []
    tried_vector = False
    if use_vector_index_first:
        try:
            emb, backend = embed_query_try_sbert(question)
            if emb is not None:
                tried_vector = True
                try:
                    candidates = retrieve_with_vector_index(emb, k=max(10, k*2))
                    diagnostics["backend"] = "vector_index"
                except Exception as e:
                    diagnostics["notes"].append(f"vector_index failed: {e}")
                    candidates = []
        except Exception as e:
            diagnostics["notes"].append(f"sbert embed failed: {e}")
            candidates = []

    # If vector path yielded nothing, use TF-IDF retrieval
    if not candidates:
        candidates = retrieve_with_tfidf_scoring(question, k=max(10, k*2))
        diagnostics["backend"] = diagnostics.get("backend") or "tfidf"

    # Re-rank top candidates by TF-IDF cross-similarity and keep top k (3 recommended for prompt)
    reranked = re_rank_by_tfidf(question, candidates, top_k=max(k, 3))
    top_k = min(k, len(reranked))
    final = reranked[:top_k]

    diagnostics["num_candidates_initial"] = len(candidates)
    diagnostics["num_final"] = len(final)
    diagnostics["re_rank_used"] = True

    # Build strict prompt using only top 3 truncated chunks
    prompt = build_strict_prompt(question, final[:3], max_chars_per_chunk=800)

    return {"retrieved": final, "prompt": prompt, "diagnostics": diagnostics}
