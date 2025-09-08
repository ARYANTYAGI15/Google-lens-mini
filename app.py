# app.py
import os
import streamlit as st

st.set_page_config(page_title="Mini Google Lens", layout="wide")
st.title("📸 Mini Google Lens with RAG")
st.write("Upload an image on the left, process it, then ask a question.")

# --- Sidebar Upload ---
uploaded_files = st.sidebar.file_uploader(
    "Upload one or more images", type=["png", "jpg", "jpeg"], accept_multiple_files=True
)
process_btn = st.sidebar.button("🚀 Process Images")
query_input = st.text_input("Ask a question about your images:")

# Lazy import (only after user interaction)
def lazy_imports():
    from modules.preprocessing import preprocessing_image
    from modules.ocr_engine import run_best_ocr
    from modules.vision_signals import extract_vision_signals
    from modules.fusion import fuse_signals
    from modules.embeddings import build_faiss_index, load_vectorstore
    from modules.rag_pipeline import rag_query
    return preprocessing_image, run_best_ocr, extract_vision_signals, fuse_signals, build_faiss_index, load_vectorstore, rag_query


# --- Processing uploaded images ---
if process_btn and uploaded_files:
    preprocessing_image, run_best_ocr, extract_vision_signals, fuse_signals, build_faiss_index, load_vectorstore, rag_query = lazy_imports()

    os.makedirs("data/uploads", exist_ok=True)

    for file in uploaded_files:
        img_path = os.path.join("data/uploads", file.name)

        # Save uploaded file
        with open(img_path, "wb") as f:
            f.write(file.read())

        # Preprocess + OCR + Vision
        versions = preprocessing_image(img_path)
        ocr_result = run_best_ocr(img_path, preprocessed_img=versions["enhanced"])
        vision_result = extract_vision_signals(img_path)

        # Fusion
        fusion_text = fuse_signals(
            image_id=os.path.splitext(file.name)[0],
            ocr_results=ocr_result["results"],
            vision_results=vision_result
        )

        st.success(f"✅ Processed {file.name}")
        with st.expander(f"Fusion result for {file.name}"):
            st.text(fusion_text)

    # Build FAISS index after all uploads
    build_faiss_index()
    st.success("🔍 FAISS index built successfully!")


# --- Query ---
if query_input:
    _, _, _, _, _, load_vectorstore, rag_query = lazy_imports()

    retriever = load_vectorstore()
    answer, retrieved = rag_query(query_input, retriever)

    st.subheader("💡 Answer")
    st.write(answer)

    st.subheader("📑 Retrieved Chunks")
    for r in retrieved:
        st.json(r)
