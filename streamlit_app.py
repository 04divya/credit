import streamlit as st
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
import numpy as np
import cv2
from sentence_transformers import SentenceTransformer, util
from ocr_similarity_utils import calculate_bert_similarity, calculate_tfidf_similarity, extract_text_from_file, classify_document

# Set UKM Theme Colors
UKM_RED = "#E60000"
UKM_BLUE = "#0066B3"
UKM_YELLOW = "#FFD700"

# Set page configuration
st.set_page_config(
    page_title="UKM Transfer Credit Checker",
    layout="centered"
)

# Title and Logo section
def display_title_and_logo():
    col1, col2 = st.columns([1, 6])
    with col1:
        st.image("https://raw.githubusercontent.com/khaliesahazmin/DataExtraction/main/logo_UKM.png", width=80)
    with col2:
        st.markdown(f"<h1 style='color:{UKM_RED};'>Transfer Credit Checker System</h1>", unsafe_allow_html=True)
        st.markdown(f"<h5 style='color:{UKM_BLUE};'>Universiti Kebangsaan Malaysia</h5>", unsafe_allow_html=True)
    st.markdown("---")

# Upload Syllabus Documents
def upload_syllabuses():
    st.markdown(f"<h3 style='color:{UKM_RED};'>📄 Upload Syllabus Documents</h3>", unsafe_allow_html=True)
    uploaded_ukm = st.file_uploader("Upload UKM Syllabus (PDF/Image)", type=['pdf', 'png', 'jpg', 'jpeg'], key="ukm_file")
    uploaded_ipts = st.file_uploader("Upload IPT Syllabuses (PDF/Image)", type=['pdf', 'png', 'jpg', 'jpeg'], accept_multiple_files=True, key="ipt_files")
    return uploaded_ukm, uploaded_ipts

# Display and process the similarity results
def display_similarity_results(ukm_text, uploaded_ipts):
    for ipt_file in uploaded_ipts:
        ipt_text = extract_text_from_file(ipt_file)
        ipt_class = classify_document(ipt_text)

        # Calculate similarity scores
        bert_score = calculate_bert_similarity(ukm_text, ipt_text)
        tfidf_score = calculate_tfidf_similarity(ukm_text, ipt_text)

        st.markdown(f"### 🏫 IPT Document: {ipt_file.name}")
        st.info(ipt_class)
        st.text_area("Extracted Text (IPT)", ipt_text, height=200)
        st.write(f"**BERT Similarity:** {bert_score:.2f}%")
        st.write(f"**TF-IDF Similarity:** {tfidf_score:.2f}%")

        # Store the similarity results in session state
        st.session_state.similarity_results.append({
            "filename": ipt_file.name,
            "bert": bert_score,
            "tfidf": tfidf_score
        })

# Main function to execute the app
def main():
    display_title_and_logo()

    # Upload files
    uploaded_ukm, uploaded_ipts = upload_syllabuses()

    # Initialize session state if not already done
    if "similarity_results" not in st.session_state:
        st.session_state.similarity_results = []

    if "reset_key" not in st.session_state:
        st.session_state.reset_key = 0

    # Run similarity comparison if both UKM and IPT files are uploaded
    if uploaded_ukm and uploaded_ipts:
        with st.spinner("🔍 Extracting and comparing..."):
            # Extract and classify UKM document
            ukm_text = extract_text_from_file(uploaded_ukm)
            ukm
