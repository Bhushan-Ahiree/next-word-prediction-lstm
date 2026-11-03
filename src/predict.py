
import json
from pathlib import Path

import tensorflow as tf


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "next_word_lstm.keras"
VOCAB_PATH = PROJECT_ROOT / "artifacts" / "vocabulary.json"

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def load_prediction_resources():
    """Load the trained model and vocabulary."""

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    with open(VOCAB_PATH, "r", encoding="utf-8") as file:
        token_to_id = json.load(file)

    id_to_token = {
        int(token_id): token
        for token, token_id in token_to_id.items()
    }

    return model, token_to_id, id_to_token


def is_displayable_word(token):
    """Check whether a token contains at least one alphanumeric character."""
    return any(character.isalnum() for character in token)


def predict_next_words(
    text,
    model,
    token_to_id,
    id_to_token,
    top_k=5,
):
    """Predict unique next-word suggestions."""

    tokens = text.split()

    context_window = model.input_shape[1]

    if len(tokens) < context_window:
        raise ValueError(
            f"Please enter at least {context_window} words."
        )

    tokens = tokens[-context_window:]

    # Convert input words into IDs.
    unknown_id = token_to_id[UNK_TOKEN]

    input_ids = [
        token_to_id.get(
            token,
            token_to_id.get(token.casefold(), unknown_id),
        )
        for token in tokens
    ]

    input_tensor = tf.constant([input_ids], dtype=tf.int32)

    # Obtain model probabilities.
    logits = model(input_tensor, training=False)[0]
    probabilities = tf.nn.softmax(logits).numpy()

    # Group case variants and exclude punctuation/special tokens.
    grouped = {}

    for token_id, token in id_to_token.items():

        if token in (PAD_TOKEN, UNK_TOKEN):
            continue

        if not is_displayable_word(token):
            continue

        probability = float(probabilities[token_id])
        key = token.casefold()

        if key not in grouped:
            grouped[key] = {
                "word": token,
                "probability": probability,
            }
        else:
            grouped[key]["probability"] += probability

            # Display the spelling with the highest individual probability.
            if probability > grouped[key].get("best_individual", 0):
                grouped[key]["word"] = token

        grouped[key]["best_individual"] = max(
            probability,
            grouped[key].get("best_individual", 0),
        )

    if not grouped:
        return []

    # Normalize across all eligible word groups.
    total_probability = sum(
        item["probability"] for item in grouped.values()
    )

    results = []

    for item in grouped.values():
        results.append({
            "word": item["word"],
            "probability": item["probability"] / total_probability,
        })

    results.sort(
        key=lambda item: item["probability"],
        reverse=True,
    )

    return results[:top_k]


def main():
    print("=" * 45)
    print("       TECH AUTOCOMPLETE - LSTM")
    print("=" * 45)

    print("\nLoading model and vocabulary...")

    model, token_to_id, id_to_token = load_prediction_resources()

    print("Model loaded successfully.")
    print("\nEnter at least 5 words to predict.")
    print("Type 'exit' to quit.")

    while True:

        text = input("\nEnter sentence: ").strip()

        if text.casefold() == "exit":
            print("Exiting prediction system.")
            break

        if not text:
            print("Please enter some text.")
            continue

        try:
            predictions = predict_next_words(
                text,
                model,
                token_to_id,
                id_to_token,
                top_k=5,
            )

            print("\nTop next-word suggestions:")

            for index, prediction in enumerate(predictions, start=1):
                print(
                    f"{index}. {prediction['word']:<20}"
                    f"{prediction['probability'] * 100:.2f}%"
                )

        except ValueError as error:
            print(f"Input error: {error}")


if __name__ == "__main__":
    main()
    