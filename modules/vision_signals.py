from ultralytics import YOLO
from pyzbar.pyzbar import decode
import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Initialize YOLO
yolo_model = YOLO("yolov8n.pt")

def detect_objects(img_path, conf_thresh=0.5):
    results = yolo_model(img_path, conf=conf_thresh)[0]  # ✅ fixed
    objects = [yolo_model.names[int(cls)] for cls in results.boxes.cls]
    return list(set(objects))

def detect_qr(img_path):
    img = cv2.imread(img_path)
    decoded = decode(img)
    return [d.data.decode("utf-8") for d in decoded]

# Initialize BLIP
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def generate_captions(img_path):
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)  # ✅ fixed
    out = model.generate(**inputs, max_new_tokens=50)
    return processor.decode(out[0], skip_special_tokens=True)

def extract_vision_signals(img_path):
    return {
        "objects": detect_objects(img_path),
        "qr_codes": detect_qr(img_path),
        "caption": generate_captions(img_path)
    }
