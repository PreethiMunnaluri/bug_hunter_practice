from sentence_transformers import SentenceTransformer
import pickle
import numpy as np

# Load embedding model locally
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load documents
with open("data.txt") as f:
    docs = f.readlines()

# Create embeddings
embeddings = model.encode(docs)
embeddings = np.array(embeddings)

# Save embeddings
with open("embeddings.pkl", "wb") as f:
    pickle.dump((docs, embeddings), f)

print("Embeddings saved locally! ✅")