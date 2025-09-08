# modules/retrieval.py
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_DIR = "data/vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_embed_model = None
_index = None
_metadatas = None

def load_index(index_dir=INDEX_DIR):
    global _embed_model, _index, _metadatas
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    if _index is None:
        _index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
    if _metadatas is None:
        with open(os.path.join(index_dir, "metadatas.json"), "r", encoding="utf-8") as f:
            _metadatas = json.load(f)

def retrieve(query, top_k=5):
    load_index()
    q_emb = _embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = _index.search(q_emb.astype("float32"), top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1: continue
        meta = _metadatas[idx]
        results.append({"score": float(score), **meta})
    return results
