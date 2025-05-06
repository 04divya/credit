from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
import numpy as np
import cv2

# Updated language setting: Malay + English
TESSERACT_LANGUAGES = 'mal+eng'

def preprocess_image(image):
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Deskewing
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Adaptive Thresholding
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Denoising
    denoised = cv2.medianBlur(binary, 3)

    return Image.fromarray(denoised)

def extract_text_from_file(file):
    text = ""
    try:
        if file.type == "application/pdf":
            images = convert_from_bytes(file.read(), dpi=500)  # High DPI
        else:
            image = Image.open(file)
            images = [image]

        for page in images:
            processed = preprocess_image(page)
            text += pytesseract.image_to_string(processed, lang=TESSERACT_LANGUAGES)
        return text
    except Exception as e:
        print(f"[OCR ERROR] {e}")
        return ""
