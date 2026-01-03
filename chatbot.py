import json
import nltk
from nltk.tokenize import word_tokenize
from unidecode import unidecode
import re
import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

nltk.download('punkt')

# ================= LOAD INTENTS =================
with open("knowledge_base.pkl", "rb") as f:
    knowledge_base = pickle.load(f)

index = faiss.read_index("store_policy_index.faiss")
# ================= LOAD SYSTEM (LAZY) =================
model = None
knowledge_base = None
index = None

def load_system():
    required_files = ['knowledge_base.pkl', 'store_policy_index.faiss', 'model_name.txt']
    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Thiếu file: {file}")

    with open('model_name.txt', 'r') as f:
        model_name = f.read().strip()

    model = SentenceTransformer(model_name)

    with open('knowledge_base.pkl', 'rb') as f:
        knowledge_base = pickle.load(f)

    index = faiss.read_index('store_policy_index.faiss')
    return model, knowledge_base, index

def ensure_loaded():
    global model, knowledge_base, index
    if model is None:
        model, knowledge_base, index = load_system()

# ================= NLP =================
def preprocess_text(text):
    return word_tokenize(unidecode(text.lower()))

def get_fallback(query):
    query_lower = query.lower()
    responses = {
        'chào': "Chào bạn! Tôi có thể giúp gì về sản phẩm hoặc chính sách?",
        'đổi': "Chính sách đổi trả trong 30 ngày nếu chưa sử dụng.",
        'giao': "Giao hàng 3–5 ngày, miễn phí cho đơn trên 500k.",
        'bảo hành': "Bảo hành 7 ngày với lỗi sản xuất.",
        'khuyến mãi': "Hiện đang có nhiều ưu đãi cuối tuần."
    }
    for k, v in responses.items():
        if k in query_lower:
            return v
    return "Tôi sẵn sàng hỗ trợ! Bạn có thể hỏi về sản phẩm hoặc chính sách."

def get_semantic_answer(query):
    ensure_loaded()
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)
    scores, indices = index.search(query_embedding.astype('float32'), 1)

    if indices[0][0] == -1 or scores[0][0] < 0.6:
        return get_fallback(query)

    match = knowledge_base['pattern_responses'][indices[0][0]]
    return match['response']

# ================= MAIN =================
def chatbot_response(user_input):
    return get_semantic_answer(user_input)

