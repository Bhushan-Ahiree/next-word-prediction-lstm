from collections import Counter
from src.preprocessing.text_preprocessing import load_split, tokenize_rows

train_df = load_split("train")
train_sequences = tokenize_rows(train_df)

token_counts = Counter(
    token
    for sequence in train_sequences
    for token in sequence
)

print("Total training tokens:", sum(token_counts.values()))
print("Unique training tokens:", len(token_counts))


print("\nTop 20 most frequent tokens:")
for token, count in token_counts.most_common(20):
    print(f"{token!r}: {count}")

print("\nVocabulary size at different frequency cutoffs:")
for cutoff in (1, 2, 5, 10):
    retained = sum(
        1 for count in token_counts.values()
        if count >= cutoff
    )
    print(f"Minimum frequency {cutoff}: {retained} tokens")

print("\nRare-token statistics:")
for threshold in (1, 2, 5):
    rare_count = sum(
        count for count in token_counts.values()
        if count <= threshold
    )
    rare_types = sum(
        1 for count in token_counts.values()
        if count <= threshold
    )
    print(
        f"Frequency <= {threshold}: "
        f"{rare_types} unique tokens, "
        f"{rare_count} total occurrences"
    )
