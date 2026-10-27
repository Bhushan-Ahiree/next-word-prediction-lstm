# Next-Word Prediction Using LSTM

## Project Overview

This project builds a word-level next-word prediction model using a Long Short-Term Memory (LSTM) neural network.

Given a sequence of words, the model will predict possible next words and return the top five candidates with their model-estimated probabilities.

The goal is to understand the complete workflow, from text preprocessing to model evaluation, rather than simply train a model.

## Objectives

- Explore and preprocess the WikiText-2 dataset.
- Convert text into word-level token IDs.
- Generate input sequences and next-word targets.
- Build an LSTM-based neural network.
- Predict the top five next-word candidates.
- Evaluate the model using loss, top-1 accuracy, and top-5 accuracy.
- Compare the model with a simple baseline.

## Dataset

- **Name:** WikiText-2
- **Configuration:** wikitext-2-v1
- **Task:** Word-level language modeling
- **Format:** Parquet
- **Source:** https://huggingface.co/datasets/Salesforce/wikitext
- **License:** CC BY-SA 4.0

The dataset contains training, validation, and test splits. The original downloaded files are stored in data/raw/.

## Planned Technology

- Python
- Pandas
- PyArrow
- TensorFlow / Keras
- Matplotlib

Additional libraries will be added only when needed.

## Planned Workflow

1. Dataset exploration
2. Text preprocessing
3. Word-level tokenization
4. Input sequence and target generation
5. Training and validation preparation
6. LSTM model construction
7. Model training
8. Evaluation and baseline comparison
9. Top-five prediction function
10. Optional interactive demo

## Evaluation

The model will be evaluated on held-out data using:

- Loss
- Top-1 accuracy
- Top-5 accuracy
- Comparison against a simple baseline

The test split will be kept separate from model training and tuning.

## Project Status

- [x] Downloaded and inspected dataset files
- [ ] Finalize preprocessing decisions
- [ ] Implement word-level tokenization
- [ ] Generate training sequences
- [ ] Build and train the LSTM
- [ ] Evaluate the model
- [ ] Implement top-five predictions
- [ ] Complete project documentation

## Scope

This project focuses on **word-level next-word prediction using an LSTM**. It is not a transformer-based language model or a code-completion system.

The model's probabilities are estimates produced by the trained model, not guarantees that a candidate is the correct next word.

