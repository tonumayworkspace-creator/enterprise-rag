import pandas as pd
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIG
# -----------------------------
TOP_K = 3
SAMPLE_SIZE = 200   # small = fast & safe

VECTOR_DB_PATH = "data/vector_db"
FAISS_INDEX_PATH = f"{VECTOR_DB_PATH}/faiss.index"
METADATA_PATH = f"{VECTOR_DB_PATH}/metadata.pkl"

QA_FILE = "data/final/enterprise_qa_pairs.csv"

# -----------------------------
# LOAD DATA
# -----------------------------
print("Loading QA pairs...")
qa_df = pd.read_csv(QA_FILE)

# Keep only valid string questions & answers
qa_df = qa_df[
    qa_df["question"].apply(lambda x: isinstance(x, str) and len(x.strip()) > 5) &
    qa_df["answer"].apply(lambda x: isinstance(x, str) and len(x.strip()) > 5)
]

# Sample for evaluation
qa_df = qa_df.sample(SAMPLE_SIZE, random_state=42)

print("Evaluation samples:", len(qa_df))

print("Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_PATH)

print("Loading metadata...")
with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# EVALUATION
# -----------------------------
hits = 0
total = 0

for _, row in qa_df.iterrows():
    query = row["question"]
    true_answer = row["answer"]

    # Embed query safely
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, TOP_K)

    retrieved_answers = []
    for idx in indices[0]:
        retrieved_answers.append(metadata[idx]["text"])

    # Check if true answer appears in retrieved answers
    if any(true_answer[:50] in ra for ra in retrieved_answers):
        hits += 1

    total += 1

precision_at_k = hits / total if total > 0 else 0.0

print("=" * 60)
print("RAG EVALUATION RESULTS")
print("Samples evaluated:", total)
print(f"Precision@{TOP_K}: {precision_at_k:.2f}")
print("=" * 60)

print("STEP 10 COMPLETED SUCCESSFULLY")
