
from pathlib import Path

import pandas as pd

path = Path("data/raw/validation-00000-of-00001.parquet")
df = pd.read_parquet(path)

unk_rows = df[
    df["text"].fillna("").str.contains("<unk>", regex=False)
]

print("Rows containing <unk>:", len(unk_rows))
print("Total <unk> occurrences:", int(
    unk_rows["text"].str.count("<unk>").sum()
))

print("\nFirst 5 examples:\n")

for text in unk_rows["text"].head(5):
    print(repr(text))
    print()
