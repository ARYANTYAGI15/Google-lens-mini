import cv2
import torch
import streamlit as st
from PIL import Image
from pyzbar.pyzbar import decode
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration


# --- Cached model loaders ---
@st.cache_resource
def get_yolo():
    return YOLO("yolov8n.pt")

@st.cache_resource
def get_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return processor, model.to(device)


# --- Vision functions ---
def detect_objects(img_path, conf_thresh=0.5):
    model = get_yolo()
    results = model(img_path, conf=conf_thresh)[0]
    objects = [model.names[int(cls)] for cls in results.boxes.cls]
    return list(set(objects))


def detect_qr(img_path):
    img = cv2.imread(img_path)
    decoded = decode(img)
    return [d.data.decode("utf-8") for d in decoded]


def generate_captions(img_path):
    processor, model = get_blip()
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=50)
    return processor.decode(out[0], skip_special_tokens=True)


def extract_vision_signals(img_path):
    return {
        "objects": detect_objects(img_path),
        "qr_codes": detect_qr(img_path),
        "caption": generate_captions(img_path)
    }
