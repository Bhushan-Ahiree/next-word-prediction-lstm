
from src.preprocessing.text_preprocessing import load_split, tokenize_rows
from src.preprocessing.vocabulary import build_vocabulary
from src.model import build_lstm_model, compile_model

CONTEXT_WINDOW = 5

# Build vocabulary from training data.
train_sequences = tokenize_rows(load_split("train"))
token_to_id, id_to_token = build_vocabulary(train_sequences)

# Build and compile model.
model = build_lstm_model(
    vocab_size=len(token_to_id),
    context_window=CONTEXT_WINDOW,
)

compile_model(model)

# Display architecture.
model.summary()

# Test one batch of two input examples.
import tensorflow as tf

sample_x = tf.constant([
    [2, 3, 4, 5, 6],
    [3, 4, 5, 6, 7],
], dtype=tf.int32)

predictions = model(sample_x)

print("\nInput shape:", sample_x.shape)
print("Output shape:", predictions.shape)
print("Expected output shape:", (2, len(token_to_id)))
