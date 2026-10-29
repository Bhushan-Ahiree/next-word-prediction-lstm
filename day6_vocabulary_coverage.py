
from collections import Counter

from src.preprocessing.text_preprocessing import load_split, tokenize_rows


MIN_FREQUENCY = 5

# Fit vocabulary statistics using training data only.
train_sequences = tokenize_rows(load_split("train"))

token_counts = Counter(
    token
    for sequence in train_sequences
    for token in sequence
)

vocabulary = {
    token
    for token, count in token_counts.items()
    if count >= MIN_FREQUENCY
}

print("Minimum frequency:", MIN_FREQUENCY)
print("Vocabulary entries retained:", len(vocabulary))
print()

# Measure coverage on each split without fitting on validation or test.
for split_name in ("train", "validation", "test"):
    sequences = tokenize_rows(load_split(split_name))

    total_tokens = 0
    retained_tokens = 0
    literal_unk_tokens = 0

    for sequence in sequences:
        for token in sequence:
            total_tokens += 1

            if token == "<unk>":
                literal_unk_tokens += 1

            if token in vocabulary:
                retained_tokens += 1

    below_cutoff_or_unseen = total_tokens - retained_tokens
    coverage = retained_tokens / total_tokens * 100

    print(f"--- {split_name.upper()} ---")
    print("Total tokens:", total_tokens)
    print("Tokens retained directly:", retained_tokens)
    print("Tokens to map to <unk>:", below_cutoff_or_unseen)
    print("Literal <unk> occurrences:", literal_unk_tokens)
    print(f"Direct vocabulary coverage: {coverage:.2f}%")
    print()
