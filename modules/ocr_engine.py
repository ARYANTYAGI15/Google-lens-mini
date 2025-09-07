import easyocr
from paddleocr import PaddleOCR

# EasyOCR init
easyocr_reader = easyocr.Reader(['en'])

# PaddleOCR init (set cls here, not in predict)
paddle_reader = PaddleOCR(use_angle_cls=True, lang='en')


def run_easyocr(img):
    results = easyocr_reader.readtext(img)
    return [
        {"text": text, "conf": float(conf), "bbox": bbox}
        for (bbox, text, conf) in results
    ]


def run_paddleocr(img_path):
    results = paddle_reader.predict(img_path)
    
    if not results:
        return []
    
    # First element contains OCR results
    ocr_data = results[0]
    
    # Some versions put results under "res"
    if "res" in ocr_data:
        parsed = [
            {"text": r["text"], "conf": float(r["confidence"]), "bbox": r["text_region"]}
            for r in ocr_data["res"]
        ]
        return parsed
    else:
        # If structure is different, just return the raw output
        return results
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
    # Run EasyOCR (if preprocessed image is provided, use it)
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