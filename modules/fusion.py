import os
import json

def fuse_signals(image_id, ocr_results, vision_results, save_dir="data/fusion_text", conf_thresh=0.5):
    """
    Combine OCR + Vision results into unified text + JSON.

    Args:
        image_id (str): unique id for the image
        ocr_results (list): list of dicts {text, conf, bbox}
        vision_results (dict): dict {objects, qr_codes, caption}
        save_dir (str): folder to save JSON
        conf_thresh (float): confidence threshold for OCR text

    Returns:
        dict with fusion_text (str) and fusion_json (dict)
    """
    # Ensure folder exists
    os.makedirs(save_dir, exist_ok=True)

    # --- Extract OCR text ---
    ocr_texts = [r["text"] for r in ocr_results if r.get("conf", 0) >= conf_thresh]
    ocr_text = " ".join(ocr_texts) if ocr_texts else "No reliable OCR text."

    # --- Vision signals ---
    objects = vision_results.get("objects", []) or []
    qr_codes = vision_results.get("qr_codes", []) or []
    caption = vision_results.get("caption", "No caption available.")

    # --- Fusion text ---
    fusion_text = f"""
    This image contains:
    - OCR text: {ocr_text}
    - Detected objects: {', '.join(objects) if objects else 'None'}
    - QR codes: {', '.join(qr_codes) if qr_codes else 'None'}
    - Caption: {caption}
    """

    # --- JSON structure ---
    fusion_json = {
        "id": image_id,
        "ocr_text": ocr_text,
        "objects": objects,
        "qr_codes": qr_codes,
        "caption": caption
    }

    # Save JSON file
    save_path = os.path.join(save_dir, f"{image_id}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(fusion_json, f, indent=4, ensure_ascii=False)

    return {"fusion_text": fusion_text.strip(), "fusion_json": fusion_json}
