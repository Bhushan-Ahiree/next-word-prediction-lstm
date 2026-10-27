
from pathlib import Path

import pandas as pd

# Load the original training split.
data_path = Path("data/raw/train-00000-of-00001.parquet")
df = pd.read_parquet(data_path)

# Keep non-empty rows for this inspection only.
non_empty_rows = df.loc[
    df["text"].notna() & df["text"].str.strip().ne(""),
    "text",
]

print("Non-empty rows:", len(non_empty_rows))
print()

# Inspect three rows without modifying the original data.
for row_number, text in enumerate(non_empty_rows.head(3), start=1):
    tokens = text.split()

    print(f"--- Example {row_number} ---")
    print("Original text:", repr(text))
    print("Tokens:", tokens)
    print("Token count:", len(tokens))
    print()
