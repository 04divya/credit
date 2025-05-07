from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load SentenceTransformer model only once
model = SentenceTransformer('models/all-mpnet-base-v2')

def calculate_bert_similarity(text1, text2):
    # Encode texts
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    # Compute cosine similarity
    similarity = util.cos_sim(emb1, emb2)
    return similarity.item() * 100  # Convert tensor to float and scale to percentage

def calculate_tfidf_similarity(text1, text2):
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    # Compute cosine similarity
    score = cosine_similarity(vectors[0], vectors[1])[0][0]
    return score * 100  # Scale to percentage
