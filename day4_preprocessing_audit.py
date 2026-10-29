
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/raw")

SPLITS = {
    "train": "train-00000-of-00001.parquet",
    "validation": "validation-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}


def audit_split(split_name, filename):
    path = DATA_DIR / filename
    df = pd.read_parquet(path)

    text_series = df["text"].fillna("").astype(str)
    non_empty = text_series[text_series.str.strip().ne("")]

    # Count whitespace-separated tokens per non-empty row.
    token_counts = non_empty.map(lambda text: len(text.split()))

    # Inspect source formatting without changing the text.
    stripped = non_empty.str.strip()
    headings = stripped[
        stripped.str.startswith("=") & stripped.str.endswith("=")
    ]

    unk_count = non_empty.map(
        lambda text: text.split().count("<unk>")
    ).sum()

    at_hyphen_count = non_empty.map(
        lambda text: text.split().count("@-@")
    ).sum()

    print(f"\n===== {split_name.upper()} =====")
    print("Total rows:", len(df))
    print("Non-empty rows:", len(non_empty))
    print("Empty/whitespace rows:", len(df) - len(non_empty))
    print("Rows matching heading pattern:", len(headings))
    print("Total whitespace-separated tokens:", int(token_counts.sum()))
    print("Median tokens per non-empty row:", float(token_counts.median()))
    print("90th percentile row token count:", float(token_counts.quantile(0.90)))
    print("Maximum tokens in a row:", int(token_counts.max()))
    print("Occurrences of <unk>:", int(unk_count))
    print("Occurrences of @-@:", int(at_hyphen_count))

    print("\nSample headings:")
    for heading in headings.head(3):
        print(repr(heading))


for split_name, filename in SPLITS.items():
    audit_split(split_name, filename)
