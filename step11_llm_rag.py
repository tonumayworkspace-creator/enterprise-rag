import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# -----------------------------
# CONFIG
# -----------------------------
DISTANCE_THRESHOLD = 1.0
TOP_K = 3

VECTOR_DB_PATH = "data/vector_db"
FAISS_INDEX_PATH = f"{VECTOR_DB_PATH}/faiss.index"
METADATA_PATH = f"{VECTOR_DB_PATH}/metadata.pkl"

# -----------------------------
# LOAD RESOURCES
# -----------------------------
print("Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_PATH)

print("Loading metadata...")
with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Initializing OpenAI client...")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# RETRIEVE CONTEXT
# -----------------------------
def retrieve_context(query):
    query_embedding = embed_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, TOP_K)

    if len(indices[0]) == 0 or distances[0][0] > DISTANCE_THRESHOLD:
        return None

    context_blocks = []
    citations = []

    for i, idx in enumerate(indices[0]):
        chunk = metadata[idx]
        context_blocks.append(chunk["text"])
        citations.append(chunk["metadata"]["answer_id"])

    return "\n".join(context_blocks), citations

# -----------------------------
# LLM GENERATION (STRICT)
# -----------------------------
def llm_answer(query):
    retrieved = retrieve_context(query)

    if retrieved is None:
        return {
            "status": "REFUSED",
            "message": "I cannot answer this question based on the available enterprise documents."
        }

    context, citations = retrieved

    system_prompt = (
        "You are an enterprise knowledge assistant.\n"
        "You must answer ONLY using the provided context.\n"
        "If the answer is not fully supported by the context, say you cannot answer.\n"
        "Do NOT use any external knowledge.\n"
    )

    user_prompt = f"""
Context:
{context}

Question:
{query}

Answer (use only the context above):
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    answer_text = response.choices[0].message.content.strip()

    return {
        "status": "ANSWERED",
        "answer": answer_text,
        "citations": citations
    }

# -----------------------------
# TEST
# -----------------------------
test_questions = [
    "My order is delayed",
    "How do I cancel my order?",
    "Who is the CEO of Google?"
]

for q in test_questions:
    print("\nQuestion:", q)
    result = llm_answer(q)

    if result["status"] == "REFUSED":
        print("Status: REFUSED")
        print(result["message"])
    else:
        print("Status: ANSWERED")
        print("Answer:", result["answer"])
        print("Citations:", result["citations"])

    print("=" * 70)

print("STEP 11 COMPLETED SUCCESSFULLY")
