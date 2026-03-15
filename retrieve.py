import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load index
index = faiss.read_index("phones.index")

# Load texts
with open("texts.json", "r") as f:
    texts = json.load(f)

def retrieve(query, top_k=3):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = [texts[i] for i in indices[0]]
    return results

# Test
if __name__ == "__main__":
    query = "phone with good battery under 500"
    results = retrieve(query)
    for r in results:
        print("-", r)
