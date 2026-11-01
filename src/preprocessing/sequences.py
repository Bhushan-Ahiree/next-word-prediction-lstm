from src.preprocessing.vocabulary import encode_sequence

def generate_next_word_examples(
    sequences,
    token_to_id,
    context_window = 5
):
    """
    Generate fixed-length input sequences and next-token targets.

    Each original text row is processed independently.
    """
    
    if context_window < 1:
        raise ValueError("context_window must be at least 1.")

    for tokens in sequences:
        
        # Convert words into integer IDs.
        encoded = encode_sequence(tokens, token_to_id)
        
        # Slide across this row without crossing its boundary.
        for i in range(len(encoded) - context_window):
            
            x = encoded[i: i + context_window]
            y = encoded[i + context_window]
            
            yield x, y