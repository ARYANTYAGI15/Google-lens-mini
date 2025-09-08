# Google Lens Mini - OCR & Vision Signals Analysis

A local implementation of a Google Lens–style system built in Python, combining **OCR, vision signal extraction, and retrieval-augmented generation (RAG)** for image understanding.  

> ⚠️ Note: Due to hardware constraints, this project uses smaller models and notebook-based testing instead of full Streamlit deployment for heavy models.

---

## Features

- **OCR Engines Integration**  
  - Tested and implemented **EasyOCR** and **PaddleOCR** for extracting text from images.  
  - Compared results to determine the most accurate engine per image.  

- **Vision Signals Extraction**  
  - Object detection with **YOLOv8 (ultralytics)**.  
  - QR code detection using **pyzbar**.  
  - Image captioning with **BLIP** from Hugging Face.  

- **Signal Fusion**  
  - Combined OCR outputs with vision signals to create a unified image understanding representation.  

- **RAG (Retrieval-Augmented Generation) Pipeline**  
  - Built an **embedding + FAISS vector store** system to enable contextual querying of images.  
  - Integrated with Hugging Face **sentence-transformers** for semantic search.  

---

## Tech Stack

- **Python Libraries**:  
  - `paddleocr`, `easyocr`, `pytesseract` (OCR)  
  - `opencv-python-headless`, `Pillow` (image processing)  
  - `torch`, `transformers`, `sentence-transformers` (vision & embeddings)  
  - `faiss-cpu` (vector store)  
  - `ultralytics` (YOLOv8), `layoutparser` (layout detection)  
  - `pyzbar` (QR code detection)  

- **Utilities**:  
  - `numpy`, `pandas`, `scikit-learn`, `rank-bm25`, `evaluate`  

- **Deployment**: Tested locally in **Jupyter Notebook** due to hardware limitations.  

---

## Limitations

- Could not use large models or full Streamlit app deployment due to GPU/CPU memory constraints.  
- Tested smaller model variants; results are acceptable but not production-grade.  
- Focused on learning, implementation, and pipeline integration rather than scaling.  

---

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/google-lens-mini.git
   cd google-lens-mini

2. Install libraries
pip install -r requirements.txt

3. Ensure system dependencies are installed (Ubuntu/Debian):

sudo apt-get install libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgl1-mesa-glx libzbar0 tesseract-ocr libgomp1

4. Run the notebook to test OCR, vision signals, and RAG pipeline:

jupyter notebook

##Key Learnings

Implemented multiple OCR engines and evaluated accuracy.

Learned how to extract vision signals (objects, QR codes, captions) and fuse them with OCR outputs.

Built a small-scale RAG pipeline with embeddings and FAISS.

Understood practical hardware limitations when deploying deep learning models in Streamlit.


###Future Improvements

Deploy full-scale models once hardware allows (larger BLIP or PaddleOCR).

Integrate fully into Streamlit for interactive image querying.

Improve fusion algorithms for better combined signal accuracy.

###Author

Your Name

GitHub: https://github.com/ARYANTYAGI15/

LinkedIn: www.linkedin.com/in/aryan-tyagi-data-scientist
