from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-mpnet-base-v2')

def calculate_bert_similarity(text1, text2):
    emb1 = model.encode(text1)
    emb2 = model.encode(text2)
    return util.cos_sim(emb1, emb2).item() * 100

def calculate_tfidf_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    score = cosine_similarity(vectors[0], vectors[1])[0][0]
    return score * 100
