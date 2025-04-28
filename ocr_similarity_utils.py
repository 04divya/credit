import json
import numpy as np
import cv2
from PIL import Image
from pdf2image import convert_from_bytes
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

def preprocess_image(image):
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    denoised = cv2.medianBlur(binary, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    sharpened = cv2.morphologyEx(denoised, cv2.MORPH_GRADIENT, kernel)
    return Image.fromarray(sharpened)

def extract_text_from_file(file):
    text = ""
    if file.type == "application/pdf":
        images = convert_from_bytes(file.read(), 400)
    else:
        image = Image.open(file)
        images = [image]

    for page in images:
        processed = preprocess_image(page)
        text += pytesseract.image_to_string(processed)
    return text

def classify_document(text):
    UKM_KEYWORDS = ["Universiti Kebangsaan Malaysia", "UKM", "Fakulti", "Program", "Kod Kursus"]
    OTHER_KEYWORDS = ["Kolej", "Politeknik", "Universiti Teknologi", "Diploma", "Institute", "Akademi", "MARA", "UITM", "UTM", "UM", "UniMAP", "Malaysian Institute"]

    ukm_hits = sum([1 for word in UKM_KEYWORDS if word.lower() in text.lower()])
    other_hits = sum([1 for word in OTHER_KEYWORDS if word.lower() in text.lower()])
    if ukm_hits > other_hits:
        return "✅ Detected as UKM Syllabus"
    elif other_hits > ukm_hits:
        return "🏫 Detected as Other Institute/Diploma Syllabus"
    else:
        return "⚠️ Institution Type Could Not Be Determined"

def calculate_bert_similarity(text1, text2):
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings1 = model.encode(text1)
    embeddings2 = model.encode(text2)
    cosine_sim = util.cos_sim(embeddings1, embeddings2)
    return round(cosine_sim.item() * 100, 2)

def calculate_tfidf_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    similarity_score = cosine_similarity(vectors[0], vectors[1])[0][0]
    return round(similarity_score * 100, 2)
