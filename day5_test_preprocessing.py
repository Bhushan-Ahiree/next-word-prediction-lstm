
from src.preprocessing.text_preprocessing import load_split, tokenize_rows


for split_name in ("train", "validation", "test"):
    df = load_split(split_name)
    sequences = tokenize_rows(df)

    token_count = sum(len(sequence) for sequence in sequences)

    print(f"\n--- {split_name.upper()} ---")
    print("Original rows:", len(df))
    print("Non-empty sequences:", len(sequences))
    print("Total tokens:", token_count)
    print("First sequence:", sequences[0][:15])

    assert len(sequences) > 0
    assert all(len(sequence) > 0 for sequence in sequences)

print("\nPreprocessing checks passed.")
