import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load phone data
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "phones_data_20250729_022344.json")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    phones = json.load(f)


# Convert phones to text chunks
texts = []

for p in phones:
    brand = p.get("brand", "Unknown brand")
    phone_model = p.get("model", "Unknown model")   # ✅ renamed
    os = p.get("operating_system", "Unknown OS")
    chipset = p.get("chipset", "Unknown chipset")
    battery = p.get("battery_capacity", "Unknown battery")
    camera = p.get("primary_camera_resolution", "Unknown camera")
    ram = p.get("ram", "Unknown RAM")
    storage = p.get("internal_storage", "Unknown storage")
    refresh_rate = p.get("refresh_rate", "Unknown refresh rate")

    text = (
        f"{brand} {phone_model} is a smartphone running {os}. "
        f"It is powered by {chipset}. "
        f"It has {ram} RAM and {storage} internal storage. "
        f"The battery capacity is {battery}. "
        f"The primary camera setup is {camera}. "
        f"The display supports a {refresh_rate} refresh rate."
    )

    texts.append(text)

# Create embeddings
embeddings = embedder.encode(texts)

# Convert to float32 (required by FAISS)
embeddings = np.array(embeddings).astype("float32")

# Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save index
faiss.write_index(index, "phones.index")

# Save text mapping
with open("texts.json", "w", encoding="utf-8") as f:
    json.dump(texts, f, indent=2)

print("FAISS index built successfully.")
