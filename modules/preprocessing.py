import cv2
import numpy as np


def preprocessing_image(img_path):
    """
    Preprocess image for OCR: grayscale, denoise, threshold, deskew, enhance.
    Returns a dict of different versions.
    """
    img = cv2.imread(img_path)
    if img is None:
     raise ValueError(f"❌ Could not load image at path: {img_path}")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    denoised = cv2.medianBlur(gray,3)

    thresh = cv2.adaptiveThreshold(
        denoised,255,cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,15,20
    )

    coords = cv2.findNonZero(thresh)
    angle = cv2.minAreaRect(coords)[-1] if coords is not None else 0
    if angle < -45:
        angle = -(90+angle)
    else:
        angle  = -angle
    (h,w) = thresh.shape[:2]
    M = cv2.getRotationMatrix2D((w//2,h//2),angle,1.0)
    deskewed = cv2.warpAffine(thresh, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    return {
        "original": img,
        "gray": gray,
        "denoised": denoised,
        "thresh": thresh,
        "deskewed": deskewed,
        "enhanced": enhanced
    }

