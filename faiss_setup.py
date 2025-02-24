import faiss
import numpy as np
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
import os

# Load raw data 
df = pd.read_csv("artifacts\customer_support_data.csv") 

if os.path.exists("artifacts\model_cache") and os.path.isdir("artifacts\model_cache"):
    model = SentenceTransformer("artifacts/model_cache/all-MiniLM-L6-v2")
# Initialize sentence transformer model 
else:
    model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="artifacts\model_cache")  # Compact and efficient

df["combined text"] = df["category"] + " " + df["intent"] + " " + df["response"]

embeddings = model.encode(df["combined text"].tolist(), normalize_embeddings=True)

# FAISS index (L2 distance-based retrieval)
DIMENSION = embeddings.shape[1]
index = faiss.IndexFlatL2(DIMENSION)
index.add(embeddings)

os.makedirs("faiss", exist_ok=True)

# Save the FAISS index and response mapping
faiss.write_index(index, r"faiss\faiss_index.bin")
with open(r"faiss\responses.pkl", "wb") as f:
    pickle.dump(df.to_dict(orient="records"), f)

print("FAISS index created and saved successfully.")