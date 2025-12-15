import json
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIG (SAFE + FAST)
# -----------------------------
MAX_CHUNKS = 50000   # reduce further if system is slow

CHUNK_FILE = "data/chunks/enterprise_chunks.json"
VECTOR_DB_PATH = "data/vector_db"

os.makedirs(VECTOR_DB_PATH, exist_ok=True)

print("Loading chunks...")
with open(CHUNK_FILE, "r", encoding="utf-8") as f:
    raw_chunks = json.load(f)

print("Total raw chunks:", len(raw_chunks))

# -----------------------------
# CLEAN & FILTER TEXT SAFELY
# -----------------------------
chunks = []
texts = []

for chunk in raw_chunks:
    text = chunk.get("text", "")

    # Keep only valid non-empty strings
    if isinstance(text, str):
        text = text.strip()
        if len(text) > 5:  # ignore very short / useless text
            chunks.append(chunk)
            texts.append(text)

# Limit chunks for speed
chunks = chunks[:MAX_CHUNKS]
texts = texts[:MAX_CHUNKS]

print("Chunks after cleaning:", len(chunks))

# -----------------------------
# LOAD EMBEDDING MODEL
# -----------------------------
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# CREATE EMBEDDINGS
# -----------------------------
print("Creating embeddings (CPU)... This may take a few minutes.")
embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True
)

embeddings = np.array(embeddings).astype("float32")

# -----------------------------
# CREATE FAISS INDEX
# -----------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

print("Adding embeddings to FAISS index...")
index.add(embeddings)

print("Total vectors in index:", index.ntotal)

# -----------------------------
# SAVE INDEX + METADATA
# -----------------------------
faiss.write_index(index, os.path.join(VECTOR_DB_PATH, "faiss.index"))

with open(os.path.join(VECTOR_DB_PATH, "metadata.pkl"), "wb") as f:
    pickle.dump(chunks, f)

print("Vector database saved successfully:")
print(" - data/vector_db/faiss.index")
print(" - data/vector_db/metadata.pkl")
print("STEP 5 COMPLETED SUCCESSFULLY")
