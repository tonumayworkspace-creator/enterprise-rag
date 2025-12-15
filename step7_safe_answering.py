import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIG
# -----------------------------
DISTANCE_THRESHOLD = 1.0   # lower = stricter
TOP_K = 3

VECTOR_DB_PATH = "data/vector_db"
FAISS_INDEX_PATH = f"{VECTOR_DB_PATH}/faiss.index"
METADATA_PATH = f"{VECTOR_DB_PATH}/metadata.pkl"

print("Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_PATH)

print("Loading metadata...")
with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

print("Total vectors in index:", index.ntotal)

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# SAFE ANSWER FUNCTION
# -----------------------------
def safe_answer(query):
    print("\nUser Query:", query)

    # Embed query
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    # Search
    distances, indices = index.search(query_embedding, TOP_K)

    # No results
    if len(indices[0]) == 0:
        return {
            "status": "REFUSED",
            "reason": "No relevant information found in knowledge base."
        }

    best_distance = distances[0][0]

    # Low confidence
    if best_distance > DISTANCE_THRESHOLD:
        return {
            "status": "REFUSED",
            "reason": "Confidence too low. Unable to answer reliably from documents."
        }

    # Safe to answer
    answers = []

    for i, idx in enumerate(indices[0]):
        chunk = metadata[idx]
        answers.append({
            "rank": i + 1,
            "answer": chunk["text"],
            "enterprise_account": chunk["metadata"]["enterprise_account"],
            "answer_id": chunk["metadata"]["answer_id"],
            "distance": float(distances[0][i])
        })

    return {
        "status": "ANSWERED",
        "answers": answers
    }

# -----------------------------
# TEST QUERIES
# -----------------------------
test_queries = [
    "My order is delayed",
    "How do I reset my password?",
    "Who is the CEO of Apple?"
]

for q in test_queries:
    result = safe_answer(q)
    print("\nResult:")
    print(result)
    print("=" * 70)

print("STEP 7 COMPLETED SUCCESSFULLY")
