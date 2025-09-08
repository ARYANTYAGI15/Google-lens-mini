import os
import json
from glob import glob
from tqdm import tqdm
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Config
INDEX_DIR = "data/vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def read_fusion_jsons(fusion_dir="data/fusion_text"):
    paths = sorted(glob(os.path.join(fusion_dir, "*.json")))
    docs = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            j = json.load(f)
        text = f"""
        Image ID: {j.get("id","")}
        OCR: {j.get("ocr_text","")}
        Objects: {", ".join(j.get("objects",[]))}
        QR: {", ".join(j.get("qr_codes",[]))}
        Caption: {j.get("caption","")}
        """
        docs.append({"id": j.get("id"), "text": text.strip(), "meta": j})
    return docs


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if end < len(text) else len(text)
    return chunks


def build_faiss_index(fusion_dir="data/fusion_text", index_dir=INDEX_DIR):
    os.makedirs(index_dir, exist_ok=True)
    model = SentenceTransformer(EMBEDDING_MODEL)

    docs = read_fusion_jsons(fusion_dir)
    all_texts, metadatas = [], []
    for d in docs:
        chunks = chunk_text(d["text"])
        for i, c in enumerate(chunks):
            all_texts.append(c)
            metadatas.append({"chunk_id": f"{d['id']}_c{i}", "image_id": d["id"], "text": c})

    embeddings = model.encode(all_texts, show_progress_bar=True, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype("float32"))

    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))
    with open(os.path.join(index_dir, "metadatas.json"), "w", encoding="utf-8") as f:
        json.dump(metadatas, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved index to {index_dir}")


def load_vectorstore(index_dir=INDEX_DIR):
    """
    Load FAISS index and metadata for querying.
    """
    index_path = os.path.join(index_dir, "faiss.index")
    meta_path = os.path.join(index_dir, "metadatas.json")

    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("❌ FAISS index or metadata not found. Run build_faiss_index() first.")

    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadatas = json.load(f)

    return index, metadatas
