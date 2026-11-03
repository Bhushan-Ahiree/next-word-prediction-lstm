
import json
from pathlib import Path

import tensorflow as tf

from src.preprocessing.text_preprocessing import (
    load_split,
    tokenize_rows,
)
from src.preprocessing.vocabulary import build_vocabulary
from src.data_pipeline import create_dataset
from src.model import build_lstm_model, compile_model


# Configuration
CONTEXT_WINDOW = 5
BATCH_SIZE = 256
EMBEDDING_DIM = 128
LSTM_UNITS = 128

# Start with a small experiment.
TRAIN_EXAMPLES = 50_000
VALIDATION_EXAMPLES = 10_000
EPOCHS = 1


def main():
    print("Loading data...")

    train_sequences = tokenize_rows(load_split("train"))
    validation_sequences = tokenize_rows(load_split("validation"))

    print("Building vocabulary...")
    token_to_id, id_to_token = build_vocabulary(train_sequences)

    vocab_size = len(token_to_id)
    print(f"Vocabulary size: {vocab_size}")

    print("Creating datasets...")

    train_dataset = create_dataset(
        train_sequences,
        token_to_id,
        context_window=CONTEXT_WINDOW,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    validation_dataset = create_dataset(
        validation_sequences,
        token_to_id,
        context_window=CONTEXT_WINDOW,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # Limit this first experiment.
    train_dataset = train_dataset.take(
        (TRAIN_EXAMPLES + BATCH_SIZE - 1) // BATCH_SIZE
    ).repeat()
    validation_dataset = validation_dataset.take(
        (VALIDATION_EXAMPLES + BATCH_SIZE - 1) // BATCH_SIZE
    )

    print("Building model...")

    model = build_lstm_model(
        vocab_size=vocab_size,
        context_window=CONTEXT_WINDOW,
        embedding_dim=EMBEDDING_DIM,
        lstm_units=LSTM_UNITS,
    )

    model = compile_model(model)

    print("Starting training...")

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        steps_per_epoch=(TRAIN_EXAMPLES + BATCH_SIZE - 1) // BATCH_SIZE,
        validation_steps=(VALIDATION_EXAMPLES + BATCH_SIZE - 1) // BATCH_SIZE,
        shuffle=False,
    )

    print("\nTraining completed.")

    for metric, value in history.history.items():
        print(f"{metric}: {value[-1]:.4f}")
        
    # Create model directory.
    MODEL_DIR = Path("artifacts")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save trained model.
    model.save(MODEL_DIR / "next_word_lstm.keras")

    # Save vocabulary.
    with open(
        MODEL_DIR / "vocabulary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            token_to_id,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nModel and vocabulary saved successfully.")


if __name__ == "__main__":
    main()
