import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Config
INDEX_DIR = "data/vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_vectorstore(index_dir=INDEX_DIR):
    """Load FAISS index and metadata."""
    index_path = os.path.join(index_dir, "faiss.index")
    meta_path = os.path.join(index_dir, "metadatas.json")

    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("❌ FAISS index or metadata not found. Process images first.")

    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadatas = json.load(f)

    return index, metadatas


def search_index(query, index, metadatas, top_k=3):
    """Retrieve top_k chunks from FAISS."""
    model = SentenceTransformer(EMBEDDING_MODEL)
    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)

    D, I = index.search(q_emb.astype("float32"), top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < len(metadatas):
            results.append({
                "score": float(score),
                "chunk_id": metadatas[idx]["chunk_id"],
                "image_id": metadatas[idx]["image_id"],
                "text": metadatas[idx]["text"]
            })
    return results


def rag_query(query, retriever, top_k=3):
    """Perform retrieval + generation."""
    index, metadatas = retriever

    # Step 1: Retrieve
    retrieved = search_index(query, index, metadatas, top_k=top_k)
    context = "\n".join(r["text"] for r in retrieved)

    # Step 2: Generate answer using Hugging Face pipeline
    generator = pipeline("text2text-generation", model="google/flan-t5-base")
    prompt = f"Answer the question using only the context.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    out = generator(prompt, max_new_tokens=200)

    answer = out[0]["generated_text"].strip()
    return answer, retrieved
