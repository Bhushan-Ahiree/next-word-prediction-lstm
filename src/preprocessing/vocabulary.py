
from collections import Counter

from src.preprocessing.text_preprocessing import load_split, tokenize_rows


MIN_FREQUENCY = 5

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def build_vocabulary(train_sequences, min_frequency=MIN_FREQUENCY):
    """
    Build a vocabulary using training sequences only.

    Special token IDs:
        <pad> = 0
        <unk> = 1
    """

    if min_frequency < 1:
        raise ValueError("min_frequency must be at least 1.")

    token_counts = Counter(
        token
        for sequence in train_sequences
        for token in sequence
    )

    # Reserve IDs for special tokens.
    token_to_id = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    # Sort by descending frequency, then alphabetically
    # to make ID assignment reproducible.
    retained_tokens = [
        token
        for token, count in token_counts.items()
        if count >= min_frequency
        and token not in (PAD_TOKEN, UNK_TOKEN)
    ]

    retained_tokens.sort(
        key=lambda token: (-token_counts[token], token)
    )

    for token in retained_tokens:
        token_to_id[token] = len(token_to_id)

    id_to_token = {
        token_id: token
        for token, token_id in token_to_id.items()
    }

    return token_to_id, id_to_token


def encode_sequence(tokens, token_to_id):
    """Convert tokens into integer IDs."""

    unknown_id = token_to_id[UNK_TOKEN]

    return [
        token_to_id.get(token, unknown_id)
        for token in tokens
    ]


if __name__ == "__main__":

    train_sequences = tokenize_rows(load_split("train"))

    token_to_id, id_to_token = build_vocabulary(
        train_sequences
    )

    print("Vocabulary size:", len(token_to_id))
    print("Padding ID:", token_to_id[PAD_TOKEN])
    print("Unknown ID:", token_to_id[UNK_TOKEN])
    print()

    print("First 10 vocabulary entries:")
    for token, token_id in list(token_to_id.items())[:10]:
        print(f"{token!r}: {token_id}")

    print()

    sample = ["the", "house", "is", "unseenword"]

    encoded = encode_sequence(sample, token_to_id)

    decoded = [
        id_to_token[token_id]
        for token_id in encoded
    ]

    print("Original tokens:", sample)
    print("Encoded IDs:", encoded)
    print("Decoded tokens:", decoded)
