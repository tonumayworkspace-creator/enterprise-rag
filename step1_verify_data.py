import pandas as pd

# Load dataset
print("Loading dataset...")
df = pd.read_csv("data/twitter_support.csv")

print("Dataset loaded successfully")
print("=" * 60)

# Basic information
print("Total rows:", len(df))
print("Columns:", df.columns.tolist())
print("=" * 60)

# Display sample rows
print("Sample data:")
print(df.head())

print("=" * 60)

# Check inbound vs outbound distribution
print("Inbound value counts:")
print(df["inbound"].value_counts())

print("=" * 60)
print("STEP 1 COMPLETED SUCCESSFULLY")
