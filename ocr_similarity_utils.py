import json
import numpy as np
import cv2
from PIL import Image
from pdf2image import convert_from_bytes
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import re
import random
from datasets import Dataset

# Load the BERT model once to avoid reloading it multiple times
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

TESSERACT_LANGUAGES = 'eng+msa'

def preprocess_image(image):
    """
    Preprocess the input image to enhance text clarity for OCR.
    This includes converting to grayscale, thresholding, denoising, and sharpening.
    """
    try:
        image = np.array(image)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        denoised = cv2.medianBlur(binary, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        sharpened = cv2.morphologyEx(denoised, cv2.MORPH_GRADIENT, kernel)
        return Image.fromarray(sharpened)
    except Exception as e:
        print(f"Error in preprocessing image: {e}")
        return image

def extract_text_from_file(file):
    """
    Extracts text from an image or PDF file.
    For PDFs, converts each page into images and processes them with OCR.
    """
    text = ""
    try:
        if file.type == "application/pdf":
            images = convert_from_bytes(file.read(), 400)
        else:
            image = Image.open(file)
            images = [image]

        for page in images:
            processed = preprocess_image(page)
            # Specify the language for OCR
            text += pytesseract.image_to_string(processed, lang=TESSERACT_LANGUAGES)
        return text
    except Exception as e:
        print(f"Error in extracting text from file: {e}")
        return ""

def extract_course_content(text):
    """
    Extracts the 'Course Content' section from the text based on predefined headings.
    Uses regular expressions for flexibility in matching headings.
    """
    start_keywords = ["Course Content", "Kandungan Kursus", "Sinopsis Kursus", "Course Synopsis"]
    end_keywords = ["Assessment", "Penilaian", "Learning Outcome", "Hasil Pembelajaran"]

    start_pattern = "|".join(start_keywords)
    end_pattern = "|".join(end_keywords)

    try:
        start_match = re.search(start_pattern, text, re.IGNORECASE)
        if not start_match:
            return "⚠️ Course Content section not found."
        
        start_idx = start_match.end()
        end_match = re.search(end_pattern, text[start_idx:], re.IGNORECASE)
        end_idx = start_idx + end_match.start() if end_match else len(text)
        
        course_content = text[start_idx:end_idx]
        return course_content.strip() if course_content.strip() else "⚠️ Course Content is empty."
    except Exception as e:
        print(f"Error in extracting course content: {e}")
        return "⚠️ Error extracting course content."

def generate_syllabus_pairs(text):
    """
    Generates pairs of syllabus content for similarity comparison.
    """
    # Extract course content
    course_content = extract_course_content(text)
    
    # Split the course content into smaller sections based on punctuation or logical parts
    sections = re.split(r'(\.|\,|\n)', course_content)
    
    # Clean and organize sections into pairs
    clean_sections = [s.strip() for s in sections if s.strip()]
    
    # Ensure there are enough sections to form pairs
    if len(clean_sections) < 2:
        return []
    
    pairs = []
    for i in range(len(clean_sections) - 1):
        for j in range(i + 1, len(clean_sections)):
            pairs.append((clean_sections[i], clean_sections[j]))
    
    return pairs

def calculate_bert_similarity(text1, text2):
    """
    Calculate similarity between two texts using BERT embeddings.
    """
    try:
        embeddings1 = model.encode(text1)
        embeddings2 = model.encode(text2)
        cosine_sim = util.cos_sim(embeddings1, embeddings2)
        return round(cosine_sim.item() * 100, 2)
    except Exception as e:
        print(f"Error in calculating BERT similarity: {e}")
        return 0.0

def calculate_tfidf_similarity(text1, text2):
    """
    Calculate similarity between two texts using TF-IDF vectorization.
    """
    try:
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([text1, text2])
        similarity_score = cosine_similarity(vectors[0], vectors[1])[0][0]
        return round(similarity_score * 100, 2)
    except Exception as e:
        print(f"Error in calculating TF-IDF similarity: {e}")
        return 0.0

def create_custom_dataset(file):
    """
    Creates a custom dataset from a PDF file with syllabus content and calculates similarities.
    """
    # Extract text from file
    text = extract_text_from_file(file)

    # Generate syllabus pairs
    pairs = generate_syllabus_pairs(text)
    
    # Create a dataset with similarities
    dataset = []
    for pair in pairs:
        # Calculate similarities using both BERT and TF-IDF
        bert_similarity = calculate_bert_similarity(pair[0], pair[1])
        tfidf_similarity = calculate_tfidf_similarity(pair[0], pair[1])

        # Simulate document similarity for fine-tuning
        dataset.append({
            "sentence1": pair[0],
            "sentence2": pair[1],
            "bert_similarity": bert_similarity,
            "tfidf_similarity": tfidf_similarity
        })
    
    return dataset
