from ultralytics import YOLO
from pyzbar.pyzbar import decode
import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
from functools import lru_cache

# --- Lazy Loaders ---
@lru_cache(maxsize=1)
def get_yolo():
    return YOLO("yolov8n.pt")

@lru_cache(maxsize=1)
def get_blip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    return processor, model, device


# --- Functions ---
def detect_objects(img_path, conf_thresh=0.5):
    yolo = get_yolo()
    results = yolo(img_path, conf=conf_thresh)[0]
    objects = [yolo.names[int(cls)] for cls in results.boxes.cls]
    return list(set(objects))


def detect_qr(img_path):
    img = cv2.imread(img_path)
    decoded = decode(img)
    return [d.data.decode("utf-8") for d in decoded]


def generate_captions(img_path):
    processor, model, device = get_blip()
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=50)
    return processor.decode(out[0], skip_special_tokens=True)


def extract_vision_signals(img_path):
    return {
        "objects": detect_objects(img_path),
        "qr_codes": detect_qr(img_path),
        "caption": generate_captions(img_path)
    }
