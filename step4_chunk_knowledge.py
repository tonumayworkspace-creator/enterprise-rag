import pandas as pd
import os
import json

# Create output folder
os.makedirs("data/chunks", exist_ok=True)

print("Loading enterprise Q-A pairs...")
qa_df = pd.read_csv("data/final/enterprise_qa_pairs.csv")

print("Total Q-A pairs:", len(qa_df))

chunks = []

print("Creating chunks...")

for idx, row in qa_df.iterrows():
    chunk = {
        "chunk_id": f"chunk_{idx}",
        "text": row["answer"],
        "metadata": {
            "question": row["question"],
            "enterprise_account": row["enterprise_account"],
            "answer_id": row["answer_id"]
        }
    }
    chunks.append(chunk)

print("Total chunks created:", len(chunks))

# Save as JSON (easy to load later)
with open("data/chunks/enterprise_chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print("Saved chunk file:")
print(" - data/chunks/enterprise_chunks.json")
print("STEP 4 COMPLETED SUCCESSFULLY")
