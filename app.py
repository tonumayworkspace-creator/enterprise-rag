import os
import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# =============================
# CONFIG
# =============================
TOP_K = 3
DISTANCE_THRESHOLD = 1.0  # lower = stricter

VECTOR_DB_PATH = "data/vector_db"
FAISS_INDEX_PATH = f"{VECTOR_DB_PATH}/faiss.index"
METADATA_PATH = f"{VECTOR_DB_PATH}/metadata.pkl"

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="📘",
    layout="centered"
)

# =============================
# CUSTOM CSS
# =============================
st.markdown("""
<style>
.chat-user {
    background-color: #2563eb;
    color: white;
    padding: 12px 16px;
    border-radius: 12px;
    margin-bottom: 8px;
    max-width: 85%;
    align-self: flex-end;
}
.chat-assistant {
    background-color: #ffffff;
    padding: 14px 16px;
    border-radius: 12px;
    margin-bottom: 8px;
    max-width: 85%;
    border-left: 4px solid #2563eb;
}
.chat-refusal {
    background-color: #fff1f2;
    padding: 14px 16px;
    border-radius: 12px;
    margin-bottom: 8px;
    max-width: 85%;
    border-left: 4px solid #dc2626;
}
.metric-box {
    background-color: #f1f5f9;
    padding: 10px;
    border-radius: 8px;
    margin-top: 8px;
    font-size: 14px;
}
.footer {
    font-size: 13px;
    color: #9ca3af;
    text-align: center;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# =============================
# LOAD RESOURCES
# =============================
@st.cache_resource
def load_resources():
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return index, metadata, embed_model, client

index, metadata, embed_model, client = load_resources()

# =============================
# SESSION STATE (CHAT HISTORY)
# =============================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =============================
# RETRIEVAL
# =============================
def retrieve_context(query):
    query_embedding = embed_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, TOP_K)

    if len(indices[0]) == 0:
        return None

    best_distance = float(distances[0][0])

    if best_distance > DISTANCE_THRESHOLD:
        return None

    context = []
    citations = []

    for idx in indices[0]:
        chunk = metadata[idx]
        context.append(chunk["text"])
        citations.append(chunk["metadata"]["answer_id"])

    return {
        "context": "\n".join(context),
        "citations": citations,
        "best_distance": best_distance
    }

# =============================
# LLM ANSWER
# =============================
def generate_answer(query):
    retrieved = retrieve_context(query)

    if retrieved is None:
        return {
            "status": "REFUSED",
            "message": "I cannot answer this question based on the available enterprise documents."
        }

    system_prompt = (
        "You are an enterprise knowledge assistant.\n"
        "Answer ONLY using the provided context.\n"
        "If the context does not fully answer the question, say you cannot answer.\n"
        "Do not use external knowledge."
    )

    user_prompt = f"""
Context:
{retrieved['context']}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    return {
        "status": "ANSWERED",
        "answer": response.choices[0].message.content.strip(),
        "citations": retrieved["citations"],
        "confidence": max(0.0, 1 - retrieved["best_distance"])
    }

# =============================
# HEADER
# =============================
st.title("📘 Enterprise Knowledge Assistant")
st.write(
    "Chat-style enterprise RAG system with hallucination control, "
    "confidence scoring, and citations."
)

# =============================
# CHAT HISTORY DISPLAY
# =============================
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
    elif msg["role"] == "assistant":
        st.markdown(f'<div class="chat-assistant">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-refusal">{msg["content"]}</div>', unsafe_allow_html=True)

# =============================
# USER INPUT
# =============================
query = st.text_input("Ask a question", placeholder="e.g. My order is delayed")

if st.button("Send"):
    if query.strip():
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": query
        })

        with st.spinner("Retrieving and reasoning..."):
            result = generate_answer(query)

        if result["status"] == "REFUSED":
            st.session_state.chat_history.append({
                "role": "refusal",
                "content": f"❌ {result['message']}"
            })
        else:
            answer_block = f"{result['answer']}"

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer_block
            })

            # Confidence + citations
            st.progress(result["confidence"])
            st.markdown(
                f"<div class='metric-box'><b>Retrieval Confidence:</b> {result['confidence']:.2f}</div>",
                unsafe_allow_html=True
            )

            st.markdown("**Citations**")
            for c in result["citations"]:
                st.markdown(f"- Source ID: `{c}`")

# =============================
# FOOTER
# =============================
st.markdown(
    "<div class='footer'>Enterprise RAG • FAISS • OpenAI • Safe & Auditable</div>",
    unsafe_allow_html=True
)
