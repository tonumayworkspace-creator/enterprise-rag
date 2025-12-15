import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------
# PATHS
# -----------------------------
VECTOR_DB_PATH = "data/vector_db"
FAISS_INDEX_PATH = f"{VECTOR_DB_PATH}/faiss.index"
METADATA_PATH = f"{VECTOR_DB_PATH}/metadata.pkl"

print("Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_PATH)

print("Loading metadata...")
with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

print("Total vectors in index:", index.ntotal)

# -----------------------------
# LOAD EMBEDDING MODEL
# -----------------------------
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# RETRIEVAL FUNCTION
# -----------------------------
def retrieve_answers(query, top_k=5):
    print("\nUser Query:", query)

    # Embed query
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    # Search FAISS
    distances, indices = index.search(query_embedding, top_k)

    results = []

    for rank, idx in enumerate(indices[0]):
        chunk = metadata[idx]
        results.append({
            "rank": rank + 1,
            "answer": chunk["text"],
            "question_context": chunk["metadata"]["question"],
            "enterprise_account": chunk["metadata"]["enterprise_account"],
            "answer_id": chunk["metadata"]["answer_id"],
            "distance": float(distances[0][rank])
        })

    return results

# -----------------------------
# TEST QUERY
# -----------------------------
test_query = "My package has not arrived yet"

results = retrieve_answers(test_query, top_k=3)

print("\nTop Retrieved Answers")
print("=" * 60)

for res in results:
    print(f"Rank {res['rank']}")
    print("Answer:", res["answer"])
    print("Enterprise Account:", res["enterprise_account"])
    print("Answer ID (citation):", res["answer_id"])
    print("Distance:", res["distance"])
    print("-" * 60)

print("STEP 6 COMPLETED SUCCESSFULLY")
