import tensorflow as tf

# Three input examples, each containing 5 token IDs.
X = [
    [2, 3, 4, 5, 6],
    [3, 4, 5, 6, 7],
    [4, 5, 6, 7, 8],
]

# One target ID for each input.
y = [7, 8, 9]

# Convert the data into a TensorFlow Dataset.
dataset = tf.data.Dataset.from_tensor_slices((X, y))

# Group examples into batches of 2.
dataset = dataset.batch(2)

# Display the batches.
for batch_x, batch_y in dataset:
    print("Input:")
    print(batch_x.numpy())

    print("Target:")
    print(batch_y.numpy())

    print("Input shape:", batch_x.shape)
    print("Target shape:", batch_y.shape)
    print("-" * 40)