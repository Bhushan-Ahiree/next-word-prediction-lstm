
from src.preprocessing.sequences import generate_next_word_examples


# Small example vocabulary
token_to_id = {
    "<pad>": 0,
    "<unk>": 1,
    "the": 2,
    "house": 3,
    "is": 4,
    "nice": 5,
    "big": 6,
}

id_to_token = {
    token_id: token
    for token, token_id in token_to_id.items()
}

tokens = ["the", "house", "is", "nice", "the", "house", "is", "big"]

examples = list(
    generate_next_word_examples(
        [tokens],
        token_to_id,
        context_window=3,
    )
)

print("Total examples:", len(examples))
print()

for x, y in examples:
    input_words = [id_to_token[token_id] for token_id in x]
    target_word = id_to_token[y]

    print("Input:", input_words)
    print("Target:", target_word)
