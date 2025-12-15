import pandas as pd
import re
import os

# Create output folder if not exists
os.makedirs("data/processed", exist_ok=True)

print("Loading raw dataset...")
df = pd.read_csv("data/twitter_support.csv")

print("Initial rows:", len(df))

# Keep only required columns
df = df[["tweet_id", "author_id", "inbound", "created_at", "text"]]

# Remove rows with missing text
df = df.dropna(subset=["text"])

# Basic text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)      # remove URLs
    text = re.sub(r"@\w+", "", text)          # remove mentions
    text = re.sub(r"#\w+", "", text)          # remove hashtags
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
    return text

print("Cleaning text...")
df["clean_text"] = df["text"].apply(clean_text)

# Separate user queries and enterprise responses
queries = df[df["inbound"] == True]
knowledge = df[df["inbound"] == False]

print("User queries:", len(queries))
print("Enterprise responses:", len(knowledge))

# Save cleaned data
queries.to_csv("data/processed/user_queries.csv", index=False)
knowledge.to_csv("data/processed/enterprise_knowledge.csv", index=False)

print("Cleaned files saved:")
print(" - data/processed/user_queries.csv")
print(" - data/processed/enterprise_knowledge.csv")
print("STEP 2 COMPLETED SUCCESSFULLY")
