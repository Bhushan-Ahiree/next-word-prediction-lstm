
import tensorflow as tf

from src.preprocessing.sequences import generate_next_word_examples


def create_dataset(
    sequences,
    token_to_id,
    context_window=5,
    batch_size=256,
    shuffle=False,
    shuffle_buffer=10000,
):
    """
    Create a batched TensorFlow dataset for next-word prediction.
    """

    def example_generator():
        yield from generate_next_word_examples(
            sequences,
            token_to_id,
            context_window=context_window,
        )

    dataset = tf.data.Dataset.from_generator(
        example_generator,
        output_signature=(
            tf.TensorSpec(
                shape=(context_window,),
                dtype=tf.int32,
            ),
            tf.TensorSpec(
                shape=(),
                dtype=tf.int32,
            ),
        ),
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=shuffle_buffer,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
