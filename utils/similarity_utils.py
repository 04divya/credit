from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def calculate_bert_similarity(text1, text2):
    try:
        embeddings1 = model.encode(text1)
        embeddings2 = model.encode(text2)
        cosine_sim = util.cos_sim(embeddings1, embeddings2)
        return round(cosine_sim.item() * 100, 2)
    except Exception as e:
        print(f"[BERT ERROR] {e}")
        return 0.0

def calculate_tfidf_similarity(text1, text2):
    try:
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([text1, text2])
        similarity_score = cosine_similarity(vectors[0], vectors[1])[0][0]
        return round(similarity_score * 100, 2)
    except Exception as e:
        print(f"[TF-IDF ERROR] {e}")
        return 0.0
