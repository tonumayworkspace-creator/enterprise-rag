import pandas as pd
import re
import os

# Create output folder
os.makedirs("data/final", exist_ok=True)

print("Loading raw dataset...")
df = pd.read_csv("data/twitter_support.csv")

print("Total rows:", len(df))

# Keep required columns INCLUDING linking column
df = df[
    [
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "in_response_to_tweet_id"
    ]
]

# Remove rows with missing text
df = df.dropna(subset=["text"])

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("Cleaning text...")
df["clean_text"] = df["text"].apply(clean_text)

# Separate questions and answers
questions = df[df["inbound"] == True]
answers = df[df["inbound"] == False]

print("Questions:", len(questions))
print("Answers:", len(answers))

# Rename columns
questions = questions.rename(columns={
    "tweet_id": "question_id",
    "clean_text": "question"
})

answers = answers.rename(columns={
    "tweet_id": "answer_id",
    "clean_text": "answer"
})

print("Linking questions to answers...")

# Link answers to questions
qa_pairs = answers.merge(
    questions,
    left_on="in_response_to_tweet_id",
    right_on="question_id",
    how="inner"
)

# Select final columns
qa_pairs = qa_pairs[
    [
        "question_id",
        "question",
        "answer_id",
        "answer",
        "author_id_x",
        "created_at_x"
    ]
]

qa_pairs = qa_pairs.rename(columns={
    "author_id_x": "enterprise_account",
    "created_at_x": "answer_time"
})

print("Total Q-A pairs created:", len(qa_pairs))

# Save output
qa_pairs.to_csv(
    "data/final/enterprise_qa_pairs.csv",
    index=False
)

print("Saved file:")
print(" - data/final/enterprise_qa_pairs.csv")
print("STEP 3 COMPLETED SUCCESSFULLY")
