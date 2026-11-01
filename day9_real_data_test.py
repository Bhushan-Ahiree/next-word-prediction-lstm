
from src.preprocessing.text_preprocessing import load_split, tokenize_rows
from src.preprocessing.vocabulary import build_vocabulary
from src.preprocessing.sequences import generate_next_word_examples


CONTEXT_WINDOW = 5

# Load training data.
train_sequences = tokenize_rows(load_split("train"))

# Build vocabulary from training data only.
token_to_id, id_to_token = build_vocabulary(train_sequences)

# Generate examples one at a time.
total_examples = 0

print("First 5 real training examples:")
print("=" * 50)

for x, y in generate_next_word_examples(
    train_sequences,
    token_to_id,
    context_window=CONTEXT_WINDOW,
):

    if total_examples < 5:
        input_words = [id_to_token[token_id] for token_id in x]
        target_word = id_to_token[y]

        print("Input words:", input_words)
        print("Input IDs:", x)
        print("Target word:", target_word)
        print("Target ID:", y)
        print()

    total_examples += 1

print("=" * 50)
print("Total generated examples:", total_examples)
print("Expected examples:", 1934654)
print("Vocabulary size:", len(token_to_id))

if total_examples == 1934654:
    print("PASS: Training example count matches Day 7.")
else:
    print("CHECK: Training example count does not match Day 7.")
