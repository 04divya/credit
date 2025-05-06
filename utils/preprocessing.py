import numpy as np
import cv2
from PIL import Image

def preprocess_image(image):
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    denoised = cv2.medianBlur(binary, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    sharpened = cv2.morphologyEx(denoised, cv2.MORPH_GRADIENT, kernel)
    return Image.fromarray(sharpened)
