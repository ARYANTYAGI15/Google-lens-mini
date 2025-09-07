import easyocr
from paddleocr import PaddleOCR
from functools import lru_cache

# --- Lazy initialization ---
@lru_cache(maxsize=1)
def get_easyocr():
    return easyocr.Reader(['en'])

@lru_cache(maxsize=1)
def get_paddle():
    return PaddleOCR(use_angle_cls=True, lang='en')


# --- OCR Functions ---
def run_easyocr(img):
    reader = get_easyocr()
    results = reader.readtext(img)
    return [
        {"text": text, "conf": float(conf), "bbox": bbox}
        for (bbox, text, conf) in results
    ]


def run_paddleocr(img_path):
    reader = get_paddle()
    results = reader.ocr(img_path)  # use .ocr for consistent output

    if not results:
        return []

    parsed = []
    for res in results[0]:  # results[0] = list of detections
        try:
            bbox, (text, conf) = res
            parsed.append({
                "text": text,
                "conf": float(conf),
                "bbox": bbox
            })
        except Exception:
            # In case structure changes
            parsed.append({
                "text": str(res),
                "conf": 0.0,
                "bbox": None
            })
    return parsed



def run_best_ocr(img_path, preprocessed_img=None):
    """
    Runs both EasyOCR and PaddleOCR, compares avg confidence,
    and returns the better result.
    
    Args:
        img_path (str): path to the image
        preprocessed_img (numpy array, optional): preprocessed version of the image for EasyOCR
    
    Returns:
        dict with keys: engine, results, avg_conf
    """
    # Run EasyOCR
    easy_results = run_easyocr(preprocessed_img if preprocessed_img is not None else img_path)
    avg_easy_conf = sum(r["conf"] for r in easy_results) / len(easy_results) if easy_results else 0

    # Run PaddleOCR
    paddle_results = run_paddleocr(img_path)
    avg_paddle_conf = sum(r["conf"] for r in paddle_results) / len(paddle_results) if paddle_results else 0

    # Decide best
    if avg_paddle_conf >= avg_easy_conf:
        return {"engine": "PaddleOCR", "results": paddle_results, "avg_conf": avg_paddle_conf}
    else:
        return {"engine": "EasyOCR", "results": easy_results, "avg_conf": avg_easy_conf}
