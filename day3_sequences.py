
# Day 3: Vocabulary and next-word sequences

text = "the house is nice the house is big"

# 1. Split the example text into tokens.
tokens = text.split()
print("Tokens:", tokens)

# 2. Build a vocabulary.
# Assign each unique token an integer ID.
vocabulary = {}

for token in tokens:
    if token not in vocabulary:
        vocabulary[token] = len(vocabulary) + 1

print("\nVocabulary:", vocabulary)

# 3. Convert tokens into integer IDs.
token_ids = [vocabulary[token] for token in tokens]
print("\nToken IDs:", token_ids)

# 4. Create training examples using 3 input tokens.
sequence_length = 3

print("\nTraining examples:")

for i in range(len(token_ids) - sequence_length):
    input_ids = token_ids[i : i + sequence_length]
    target_id = token_ids[i + sequence_length]

    print("Input IDs:", input_ids, "Target ID:", target_id)

# 5. Display the words represented by each example.
id_to_word = {token_id: token for token, token_id in vocabulary.items()}

print("\nExamples in words:")

for i in range(len(token_ids) - sequence_length):
    input_ids = token_ids[i : i + sequence_length]
    target_id = token_ids[i + sequence_length]

    input_words = [id_to_word[token_id] for token_id in input_ids]
    target_word = id_to_word[target_id]

    print(f"{input_words} -> {target_word}")
