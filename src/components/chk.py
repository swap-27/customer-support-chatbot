from sentence_transformers import SentenceTransformer

# Load and save the model properly
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model.save("artifacts/model_cache/all-MiniLM-L6-v2")

print("Model downloaded and saved successfully.")