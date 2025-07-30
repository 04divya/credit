from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
import numpy as np
import cv2
from .preprocessing import preprocess_image

TESSERACT_LANGUAGES = 'mal+eng'

def extract_text_from_file(file):
    text = ""
    try:
        if file.type == "application/pdf":
            images = convert_from_bytes(file.read(), 500, first_page=1, last_page=3)  # Max 3 pages

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
