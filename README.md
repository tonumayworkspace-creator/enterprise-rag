![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![RAG](https://img.shields.io/badge/RAG-Enterprise--Grade-green)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-black?logo=openai)
![FAISS](https://img.shields.io/badge/Vector%20DB-FAISS-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?logo=streamlit)
![Status](https://img.shields.io/badge/Status-Production--Ready-success)

# 📘 Enterprise Knowledge Assistant (RAG)
An **enterprise-grade Retrieval-Augmented Generation (RAG) system** that delivers **accurate, source-grounded answers** over internal knowledge with **hallucination control, refusal logic, evaluation metrics, and a production-style UI**.

This project is designed to reflect **real-world GenAI systems** used in companies — not a tutorial demo.

---

## 🚀 Project Overview

Most RAG demos stop at *“chat with documents”*.  
This system goes further by addressing **enterprise AI challenges**:

- ❌ Hallucinated answers  
- ❌ No confidence or refusal logic  
- ❌ No evaluation of retrieval quality  
- ❌ Unsafe answers when data is missing  

✅ This assistant **answers only when evidence exists** and **refuses otherwise**.

---

## 🧠 Key Features

### 🔍 Grounded Retrieval
- Semantic search using **FAISS**
- Sentence embeddings via **SentenceTransformers (MiniLM)**
- Metadata-aware retrieval for traceability

### 🛑 Hallucination Control
- Confidence-based refusal logic
- Distance threshold gating
- Minimum context validation

### 🧾 Citation Enforcement
- Every answer is backed by source IDs
- No external knowledge leakage

### 📊 RAG Evaluation
- Precision@K on real question–answer pairs
- Answer vs refusal rate analysis
- Retrieval quality inspection

### 💬 Enterprise-Style UI
- Chat-based interface
- Session memory
- Visual retrieval confidence indicator

---

## 🏗️ System Architecture

```
User Query
   ↓
Text Embedding (MiniLM)
   ↓
FAISS Vector Search
   ↓
Confidence & Safety Gate
   ↓
LLM (OpenAI) – Context Only
   ↓
Answer + Citations
   ↓
Streamlit UI
```

---

## 🧰 Tech Stack

| Layer | Technology |
|-----|-----------|
| Language | Python |
| Embeddings | SentenceTransformers (MiniLM) |
| Vector Database | FAISS |
| LLM | OpenAI (GPT-4o-mini) |
| UI | Streamlit |
| Evaluation | Precision@K (custom) |
| Dataset | Kaggle – Customer Support on Twitter |

---

## 📂 Project Structure

```
enterprise-rag/
├── data/
│   ├── twitter_support.csv
│   ├── processed/
│   ├── final/
│   ├── chunks/
│   └── vector_db/
├── ingestion/
├── chunking/
├── retrieval/
├── generation/
├── evaluation/
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run Locally

### 1️⃣ Clone the Repository
```bash
git clone <your-repo-url>
cd enterprise-rag
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Set OpenAI API Key
```bash
setx OPENAI_API_KEY "your_real_api_key_here"
```
> Restart the terminal after setting the key.

### 5️⃣ Run the Application
```bash
streamlit run app.py
```

---

## 🛡️ Safety & Governance

- ✅ Refuses to answer when evidence is insufficient
- ✅ Answers strictly constrained to retrieved documents
- ✅ Citation enforcement for every response
- ✅ Configurable thresholds via environment variables
- ✅ Secrets handled via environment variables (not code)

---

## 📈 Evaluation Summary

- **Precision@K** computed on real customer-support Q–A pairs
- Demonstrates retrieval quality under partial index coverage
- Reflects real-world RAG tradeoffs and limitations

---

## 💼 Interview-Ready Explanation

> “I built an enterprise RAG system that retrieves answers strictly from internal documents, enforces confidence-based refusal to prevent hallucinations, provides citations, and evaluates retrieval quality using Precision@K.”

This project demonstrates:
- Applied AI engineering
- GenAI safety practices
- Retrieval evaluation
- Production system thinking

---

## 🎯 Target Roles

- Data Scientist (Applied AI)
- GenAI / LLM Engineer (Junior–Mid)
- Analytics Engineer (AI-enabled)
- Enterprise Search / Knowledge Systems Engineer

---

## 🔮 Future Enhancements

- Hybrid search (BM25 + Vector)
- Role-based access control
- Model drift detection
- Evaluation dashboard
- Multi-source ingestion (PDFs, Wikis, Tickets)

---

## 📌 Disclaimer

This project uses **publicly available data** for educational and portfolio purposes.  
No proprietary or sensitive enterprise data is included.

---

## 🙌 Author

**Tonumay Bhattacharya**  
Aspiring Data Scientist & GenAI Engineer
