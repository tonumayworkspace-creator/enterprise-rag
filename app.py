import os
import pickle
import streamlit as st
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# =============================
# CONFIG
# =============================
DATA_PATH = "data/twitter_support.csv"
VECTOR_DB_PATH = "data/vector_db"
FAISS_INDEX_PATH = f"{VECTOR_DB_PATH}/faiss.index"
METADATA_PATH = f"{VECTOR_DB_PATH}/metadata.pkl"

TOP_K = 3
DISTANCE_THRESHOLD = 1.0

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="📘",
    layout="centered"
)

# =============================
# BUILD VECTOR DB (IF MISSING)
# =============================
def build_vector_db():
    st.info("Building vector database (first run)…")

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text"])

    texts = df["text"].astype(str).tolist()

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    metadata = [{"text": t} for t in texts]

    os.makedirs(VECTOR_DB_PATH, exist_ok=True)

    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    return index, metadata, model

# =============================
# LOAD RESOURCES
# =============================
@st.cache_resource
def load_resources():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if not os.path.exists(FAISS_INDEX_PATH):
        index, metadata, embed_model = build_vector_db()
    else:
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)

    return index, metadata, embed_model, client

index, metadata, embed_model, client = load_resources()

# =============================
# RETRIEVAL
# =============================
def retrieve_context(query):
    q_emb = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(q_emb, TOP_K)

    if distances[0][0] > DISTANCE_THRESHOLD:
        return None

    context = [metadata[i]["text"] for i in indices[0]]
    return "\n".join(context), float(distances[0][0])

# =============================
# UI
# =============================
st.title("📘 Enterprise Knowledge Assistant")
query = st.text_input("Ask a question")

if st.button("Ask"):
    if query:
        with st.spinner("Retrieving knowledge…"):
            result = retrieve_context(query)

        if result is None:
            st.error("❌ Unable to answer reliably from available data.")
        else:
            context, distance = result

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Answer strictly from context."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
                ],
                temperature=0
            )

            st.success(response.choices[0].message.content)
            st.progress(max(0.0, 1 - distance))
