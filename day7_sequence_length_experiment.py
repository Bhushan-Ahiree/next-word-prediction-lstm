
from src.preprocessing.text_preprocessing import load_split, tokenize_rows


CONTEXT_WINDOWS = [5, 10, 20]

train_sequences = tokenize_rows(load_split("train"))

print("TRAINING SEQUENCE-LENGTH EXPERIMENT")
print("=" * 45)

total_rows = len(train_sequences)
total_tokens = sum(len(sequence) for sequence in train_sequences)

print("Non-empty text rows:", total_rows)
print("Total tokens:", total_tokens)
print()

for window_size in CONTEXT_WINDOWS:

    total_examples = 0
    contributing_rows = 0

    for sequence in train_sequences:

        # A row needs at least window_size + 1 tokens.
        examples_in_row = max(0, len(sequence) - window_size)

        total_examples += examples_in_row

        if examples_in_row > 0:
            contributing_rows += 1

    print(f"--- Context window: {window_size} ---")
    print("Training examples:", total_examples)
    print("Rows contributing:", contributing_rows)
    print(
        "Rows contributing (%):",
        f"{contributing_rows / total_rows * 100:.2f}%"
    )
    print()
