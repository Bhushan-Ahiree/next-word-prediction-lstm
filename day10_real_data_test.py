
from src.preprocessing.text_preprocessing import load_split, tokenize_rows
from src.preprocessing.vocabulary import build_vocabulary
from src.data_pipeline import create_dataset

CONTEXT_WINDOW = 5
BATCH_SIZE = 256

# Load and preprocess training data.
train_sequences = tokenize_rows(load_split("train"))

# Build vocabulary using training data only.
token_to_id, id_to_token = build_vocabulary(train_sequences)

# Create TensorFlow dataset.
train_dataset = create_dataset(
    train_sequences,
    token_to_id,
    context_window=CONTEXT_WINDOW,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

print("Vocabulary size:", len(token_to_id))
print("Batch size:", BATCH_SIZE)
print()

# Inspect first two batches.
for batch_number, (batch_x, batch_y) in enumerate(train_dataset.take(2), start=1):
    print(f"Batch {batch_number}")
    print("Input shape:", batch_x.shape)
    print("Target shape:", batch_y.shape)
    print("Input data type:", batch_x.dtype)
    print("Target data type:", batch_y.dtype)

    # Decode one example for understanding.
    input_ids = batch_x[0].numpy()
    target_id = batch_y[0].numpy()

    input_words = [id_to_token[int(token_id)] for token_id in input_ids]

    print("Sample input:", input_words)
    print("Sample target:", id_to_token[int(target_id)])
    print("-" * 40)
