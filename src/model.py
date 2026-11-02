import tensorflow as tf

def build_lstm_model(
    vocab_size,
    context_window=5,
    embedding_dim=128,
    lstm_units=128,
):
    model = tf.keras.Sequential([
        # Input: sequence of 5 integer token IDs.
        tf.keras.layers.Input(shape=(context_window,)),

        # Convert each token ID into a trainable vector.
        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
        ),

        # Learn patterns from the sequence of word vectors.
        tf.keras.layers.LSTM(lstm_units),

        # Produce one score for every vocabulary token.
        tf.keras.layers.Dense(vocab_size),
    ])
    
    return model

def compile_model(model):
    model.compile(
        optimizer = tf.keras.optimizers.Adam(),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        ),
        metrics = ["accuracy"],
    )
    
    return model