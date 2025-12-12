# main code for retrieval-augmented generation (RAG) functionality

import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from sentence_transformers import SentenceTransformer
import faiss

# Lazy-loaded embedding model
_EMBED_MODEL = None

def get_embed_model(name: str = "all-MiniLM-L6-v2"):
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer(name)
    return _EMBED_MODEL

def load_meta(meta_path: str = "data/meta.jsonl") -> List[Dict[str, Any]]:
    p = Path(meta_path)
    if not p.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    metas = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            metas.append(json.loads(line))
    return metas

def load_faiss(index_path: str = "data/faiss.index"):
    p = Path(index_path)
    if not p.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    idx = faiss.read_index(str(p))
    return idx

def embed_query(query: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    model = get_embed_model(model_name)
    emb = model.encode([query], show_progress_bar=False, normalize_embeddings=True)
    return np.array(emb, dtype="float32")

def _normalize_scores(D: np.ndarray) -> np.ndarray:
    """
    Convert FAISS returned D values into similarity scores in range roughly [-1,1] where bigger = better.
    If D looks like distances (large positive values), convert via sim = 1 - D (assumes D was 1 - cos).
    If D looks like inner-product / cosine similarities (in [-1,1]), return as-is.
    """
    if D.size == 0:
        return D
    d_min = float(np.min(D))
    d_max = float(np.max(D))
    # Heuristic: if all values are within [-1.1, 1.1], treat as similarity (inner product / cosine)
    if d_min >= -1.1 and d_max <= 1.1:
        return D
    # Otherwise treat as distance-like (bigger is worse). Convert to similarity:
    # This is a heuristic: map sim = 1 - D (so smaller distance -> higher sim).
    # Clamp output to reasonable bounds.
    sims = 1.0 - D
    return sims

def retrieve(question: str, k: int = 5, model_name: str = "all-MiniLM-L6-v2",
             meta_path: str = "data/meta.jsonl", index_path: str = "data/faiss.index",
             similarity_threshold: float = 0.20) -> List[Dict[str, Any]]:
    """
    Retrieve top-k relevant chunks and filter by similarity threshold.
    Returns a list of meta dicts (with added '_score' and '_index_pos').
    """
    metas = load_meta(meta_path)
    idx = load_faiss(index_path)

    q_emb = embed_query(question, model_name)  # shape (1, d)
    D, I = idx.search(q_emb, k)  # D shape (1,k), I shape (1,k)
    D = np.array(D, dtype=float)
    I = np.array(I, dtype=int)

    # Convert returned D to similarity scores (bigger = better)
    sims = _normalize_scores(D[0])  # 1-D array length k

    retrieved = []
    for rank, idx_pos in enumerate(I[0].tolist()):
        if idx_pos < 0 or idx_pos >= len(metas):
            continue
        sim_score = float(sims[rank])
        # Filter by threshold
        if sim_score < similarity_threshold:
            # skip low-sim matches
            continue
        m = metas[idx_pos].copy()
        m["_score"] = sim_score
        m["_index_pos"] = int(idx_pos)
        retrieved.append(m)

    return retrieved

def build_prompt(retrieved: List[Dict[str, Any]], question: str, max_ctx_chars: int = 3000) -> str:
    """
    Build a grounded prompt. If no retrieved contexts, return a short prompt that instructs model to say "I don't know."
    This prompt instructs the LLM to:
      1) Begin with a single, direct sentence that answers the question (numbers first if applicable).
      2) Then give a short justification (1-3 sentences).
      3) End with a 'Sources:' line listing page numbers.
    """
    if not retrieved:
        return (
            "You are an assistant that must answer using ONLY the provided document excerpts.\n"
            "No relevant excerpts were found for the user's question. Therefore, reply exactly: I don't know.\n\n"
            f"QUESTION:\n{question}\n\nAnswer:"
        )

    instruction = (
        "You are an assistant that must answer using ONLY the provided document excerpts.\n"
        "INSTRUCTIONS:\n"
        "1) **Start your response with one short direct sentence that answers the question.** If the answer includes numeric projections, put the numbers first (for example: 'Real GDP growth is projected to improve to 2 percent in 2024–25.').\n"
        "2) Then provide a brief justification in 1-3 sentences using only the content shown in Context.\n"
        "3) Finish with a 'Sources:' line listing the page numbers cited (e.g., 'Sources: page 2').\n"
        "If the answer is not present in the excerpts, reply exactly: I don't know.\n\n"
    )

    ctxs = []
    used_chars = 0
    for r in retrieved:
        txt = (r.get("text") or "").strip()
        if not txt:
            continue
        entry = f"[page {r.get('page')}] {txt}"
        entry_len = len(entry)
        if used_chars + entry_len > max_ctx_chars and used_chars > 0:
            break
        ctxs.append(entry)
        used_chars += entry_len

    context_block = "\n\n".join(ctxs)
    prompt = f"{instruction}Context:\n{context_block}\n\nQUESTION:\n{question}\n\nAnswer:"
    return prompt

    instruction = (
        "You are an assistant that must answer using ONLY the provided document excerpts.\n"
        "If the answer is not present in the excerpts, reply exactly: I don't know.\n"
        "Answer concisely and cite page numbers in the 'Sources:' section at the end.\n\n"
    )

    ctxs = []
    used_chars = 0
    for r in retrieved:
        txt = (r.get("text") or "").strip()
        if not txt:
            continue
        entry = f"[page {r.get('page')}] {txt}"
        entry_len = len(entry)
        if used_chars + entry_len > max_ctx_chars and used_chars > 0:
            break
        ctxs.append(entry)
        used_chars += entry_len

    context_block = "\n\n".join(ctxs)
    prompt = f"{instruction}Context:\n{context_block}\n\nQUESTION:\n{question}\n\nAnswer:"
    return prompt

def retrieve_and_prepare(question: str, k: int = 5, **kwargs) -> Dict[str, Any]:
    """
    High-level helper used by the app.
    Returns:
      {
        "retrieved": [ ... ],
        "prompt": "<string>",
      }
    """
    # allow caller to override similarity_threshold via kwargs if needed
    similarity_threshold = kwargs.pop("similarity_threshold", 0.20)
    retrieved = retrieve(question, k=k, similarity_threshold=similarity_threshold, **kwargs)
    prompt = build_prompt(retrieved, question)
    return {"retrieved": retrieved, "prompt": prompt}
