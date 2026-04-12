"""
Embeddings utility for stock FAQ data.
Builds a FAISS vector index from stock_faq.csv for similarity search.
"""
import os
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd

# Resolve paths relative to this script's location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Load data
csv_path = os.path.join(DATA_DIR, "stock_faq.csv")
data = pd.read_csv(csv_path)
questions = data["question"].tolist()
answers = data["answer"].tolist()

# Embed questions
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(questions)

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save index and data
index_path = os.path.join(DATA_DIR, "stock_faq.index")
pkl_path = os.path.join(DATA_DIR, "stock_faq_data.pkl")

faiss.write_index(index, index_path)
pd.to_pickle({"questions": questions, "answers": answers}, pkl_path)

print(f"Index saved to: {index_path}")
print(f"Data saved to: {pkl_path}")