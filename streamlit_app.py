import streamlit as st
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
import numpy as np
import cv2
import json
from sentence_transformers import SentenceTransformer, util
from your_module_name import classify_document

# Set UKM Theme Colors
UKM_RED = "#E60000"
UKM_BLUE = "#0066B3"
UKM_YELLOW = "#FFD700"

st.set_page_config(
    page_title="UKM Transfer Credit Checker",
    layout="centered"
)

# Title and Logo
col1, col2 = st.columns([1, 6])
with col1:
    st.image("https://raw.githubusercontent.com/khaliesahazmin/DataExtraction/main/logo_UKM.png", width=80)
with col2:
    st.markdown(f"<h1 style='color:{UKM_RED};'>Transfer Credit Checker System</h1>", unsafe_allow_html=True)
    st.markdown(f"<h5 style='color:{UKM_BLUE};'>Universiti Kebangsaan Malaysia</h5>", unsafe_allow_html=True)

st.markdown("---")

# Upload files
st.markdown(f"<h3 style='color:{UKM_RED};'>📄 Upload Syllabus Documents</h3>", unsafe_allow_html=True)
uploaded_ukm = st.file_uploader("Upload UKM Syllabus (PDF/Image)", type=['pdf', 'png', 'jpg', 'jpeg'], key="ukm_file")
uploaded_ipts = st.file_uploader("Upload IPT Syllabuses (PDF/Image)", type=['pdf', 'png', 'jpg', 'jpeg'], accept_multiple_files=True, key="ipt_files")

# State initialization
if "similarity_results" not in st.session_state:
    st.session_state.similarity_results = []

if "reset_key" not in st.session_state:
    st.session_state.reset_key = 0

# Function to extract text from PDF or image file
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
        return "⚠️ Error processing the file."


# Handle file uploads safely
if uploaded_ukm and uploaded_ipts:
    try:
        with st.spinner("🔍 Extracting and comparing..."):
            ukm_text = extract_text_from_file(uploaded_ukm)
            ukm_class = classify_document(ukm_text)

            st.markdown("### 📘 UKM Syllabus Document")
            st.info(ukm_class)
            st.text_area("Extracted Text (UKM)", ukm_text, height=200)

            for ipt_file in uploaded_ipts:
                ipt_text = extract_text_from_file(ipt_file)
                ipt_class = classify_document(ipt_text)

                bert_score = calculate_bert_similarity(ukm_text, ipt_text)
                tfidf_score = calculate_tfidf_similarity(ukm_text, ipt_text)

                st.markdown(f"### 🏫 IPT Document: {ipt_file.name}")
                st.info(ipt_class)
                st.text_area("Extracted Text (IPT)", ipt_text, height=200)
                st.write(f"**BERT Similarity:** {bert_score:.2f}%")
                st.write(f"**TF-IDF Similarity:** {tfidf_score:.2f}%")

                st.session_state.similarity_results.append({
                    "filename": ipt_file.name,
                    "bert": bert_score,
                    "tfidf": tfidf_score
                })
    except Exception as e:
        st.error(f"An error occurred during file processing: {e}")

# Reset and rerun button
st.markdown("---")
if st.button("🔁 Next Course / Reset"):
    st.session_state.similarity_results = []
    st.session_state.reset_key += 1
    st.rerun()


# Footer
st.markdown("---")
st.markdown(f"<p style='text-align:center;color:{UKM_BLUE};'>© 2025 Universiti Kebangsaan Malaysia | Transfer Credit Checker</p>", unsafe_allow_html=True)
