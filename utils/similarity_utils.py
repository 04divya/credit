from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load SentenceTransformer model only once:
model = SentenceTransformer('all-mpnet-base-v2')  # fallback if local model not found

def calculate_bert_similarity(text1, text2):
    """Calculate cosine similarity using BERT sentence embeddings."""
    if not text1.strip() or not text2.strip():
        return 0.0  # Avoid empty input error
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    similarity = util.cos_sim(emb1, emb2)
    return float(similarity.item()) * 100

def calculate_tfidf_similarity(text1, text2):
    """Calculate cosine similarity using TF-IDF vectorization."""
    if not text1.strip() or not text2.strip():
        return 0.0
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    score = cosine_similarity(vectors[0], vectors[1])[0][0]
    return float(score) * 100
