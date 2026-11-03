
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

import string


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "next_word_lstm.keras"
VOCAB_PATH = PROJECT_ROOT / "artifacts" / "vocabulary.json"


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
    """Exclude punctuation-only tokens from suggestions."""
    return any(character.isalnum() for character in token)


def predict_next_words(text, model, token_to_id, id_to_token, top_k=5):
    """Predict the top-k most likely next words."""

    tokens = text.split()

    if len(tokens) == 0:
        raise ValueError("Please enter some text.")

    context_window = model.input_shape[1]

    if len(tokens) < context_window:
        raise ValueError(
            f"Please enter at least {context_window} words."
        )

    # Use only the most recent context words.
    tokens = tokens[-context_window:]

    # Convert words into IDs. Unknown words become <unk>.
    input_ids = [
        token_to_id.get(token, token_to_id["<unk>"])
        for token in tokens
    ]

    input_tensor = tf.constant(
        [input_ids],
        dtype=tf.int32,
    )

    # Get model logits.
    logits = model(input_tensor, training=False)[0]

    # Do not suggest special tokens.
    logits = tf.tensor_scatter_nd_update(
        logits,
        indices=[[token_to_id["<pad>"]], [token_to_id["<unk>"]]],
        updates=[-1e9, -1e9],
    )

    # Convert logits into probabilities.
    probabilities = tf.nn.softmax(logits)

    # Get the highest-probability words.
    
    # Filter punctuation-only tokens.
    valid_ids = [
        token_id
        for token_id, token in id_to_token.items()
        if is_displayable_word(token)
    ]

    valid_logits = tf.gather(probabilities, valid_ids)

    top_probabilities, top_positions = tf.math.top_k(
        valid_logits,
        k=top_k,
    )

    top_ids = tf.gather(
        tf.constant(valid_ids, dtype=tf.int32),
        top_positions,
    )

    results = []

    for probability, token_id in zip(
        top_probabilities.numpy(),
        top_ids.numpy(),
    ):
        results.append({
            "word": id_to_token[int(token_id)],
            "probability": float(probability),
        })

    return results


if __name__ == "__main__":
    model, token_to_id, id_to_token = load_prediction_resources()

    text = input("Enter at least 5 words: ")

    predictions = predict_next_words(
        text,
        model,
        token_to_id,
        id_to_token,
        top_k=5,
    )

    print("\nTop 5 next-word suggestions:")

    for index, prediction in enumerate(predictions, start=1):
        print(
            f"{index}. {prediction['word']} "
            f"- {prediction['probability'] * 100:.2f}%"
        )
