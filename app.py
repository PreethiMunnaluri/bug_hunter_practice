from flask import Flask, render_template, request
import pickle
import numpy as np
import os

from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

app = Flask(__name__)

# -----------------------------
# Load Gemini API Key
# -----------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not set")

# -----------------------------
# Configure Gemini
# -----------------------------
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-flash-latest")

# -----------------------------
# Load RAG Embeddings (Pickle)
# -----------------------------
with open("embeddings.pkl", "rb") as f:
    docs, embeddings = pickle.load(f)

# -----------------------------
# RAG Search (Cosine Similarity)
# -----------------------------
def search_rag(query):
    # Encode query as a lightweight vector
    q_vec = np.array([hash(query) % 1000] * embeddings.shape[1]).reshape(1, -1)
    
    sims = cosine_similarity(q_vec, embeddings)
    idx = sims.argmax()
    return docs[idx]

# -----------------------------
# Call Gemini API
# -----------------------------
def call_llm(prompt):
    response = model.generate_content(prompt)
    return response.text

# -----------------------------
# Flask Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        code = request.form["code"]
        context = search_rag(code)
        prompt = f"""
You are an AI bug hunter.

Use this context:
{context}

Analyze this code:
{code}

Explain bugs and fixes in simple words.
"""
        try:
            result = call_llm(prompt)
        except Exception as e:
            result = f"Error from Gemini API: {str(e)}"

    return render_template("index.html", result=result)

# -----------------------------
# Run App (Railway Compatible)
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
