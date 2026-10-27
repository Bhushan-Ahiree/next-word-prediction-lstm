
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")

files = {
    "train": "train-00000-of-00001.parquet",
    "validation": "validation-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}

for split_name, filename in files.items():
    file_path = DATA_DIR / filename

    print(f"\n--- {split_name.upper()} ---")

    if not file_path.exists():
        print(f"File not found: {file_path}")
        continue

    df = pd.read_parquet(file_path)

    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))
  
# Day 1: inspect empty rows and text boundaries

print("\n=== EMPTY ROW COUNTS ===")

for split_name, filename in files.items():
    df = pd.read_parquet(DATA_DIR / filename)

    empty_count = df["text"].fillna("").str.strip().eq("").sum()

    print(f"{split_name}: {empty_count} empty rows out of {len(df)}")


print("\n=== FIRST 20 TRAINING ROWS ===")

train_df = pd.read_parquet(DATA_DIR / files["train"])

for i in range(min(20, len(train_df))):
    print(f"Row {i}: {repr(train_df.loc[i, 'text'])}")


print("\n=== NEWLINE REPRESENTATION ===")

sample = train_df.loc[4, "text"]

print("Actual newline characters:", sample.count("\n"))
print("Literal backslash-n sequences:", sample.count("\\n"))
