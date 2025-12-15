import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIG
# -----------------------------
DISTANCE_THRESHOLD = 1.0
TOP_K = 3

VECTOR_DB_PATH = "data/vector_db"
FAISS_INDEX_PATH = f"{VECTOR_DB_PATH}/faiss.index"
METADATA_PATH = f"{VECTOR_DB_PATH}/metadata.pkl"

print("Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_PATH)

print("Loading metadata...")
with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# RETRIEVE + SAFE CHECK
# -----------------------------
def retrieve_safe_context(query):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, TOP_K)

    if len(indices[0]) == 0 or distances[0][0] > DISTANCE_THRESHOLD:
        return None

    context_blocks = []

    for i, idx in enumerate(indices[0]):
        chunk = metadata[idx]
        context_blocks.append({
            "text": chunk["text"],
            "source": chunk["metadata"]["answer_id"],
            "enterprise_account": chunk["metadata"]["enterprise_account"],
            "distance": float(distances[0][i])
        })

    return context_blocks

# -----------------------------
# GENERATE ANSWER (SAFE)
# -----------------------------
def generate_answer(query):
    context = retrieve_safe_context(query)

    if context is None:
        return {
            "status": "REFUSED",
            "message": "I’m unable to answer this question based on the available enterprise documents."
        }

    # Combine answers (no hallucination)
    answer_text = "Based on our support knowledge:\n\n"

    for block in context:
        answer_text += f"- {block['text']}\n"

    citations = [block["source"] for block in context]

    return {
        "status": "ANSWERED",
        "answer": answer_text,
        "citations": citations
    }

# -----------------------------
# TEST QUERIES
# -----------------------------
test_queries = [
    "My order is delayed",
    "How do I cancel my order?",
    "Who invented Python?"
]

for q in test_queries:
    print("\nUser Question:", q)
    result = generate_answer(q)

    if result["status"] == "REFUSED":
        print("Status: REFUSED")
        print(result["message"])
    else:
        print("Status: ANSWERED")
        print(result["answer"])
        print("Citations:", result["citations"])

    print("=" * 70)

print("STEP 8 COMPLETED SUCCESSFULLY")
