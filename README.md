# AspectSense — Aspect-Based Sentiment Analysis Using Deep Learning

A simple NLP lab project that analyzes electronic-product reviews for four predefined aspects: **Camera, Battery, Display, and Price**. For every detected aspect, a **BiLSTM** model predicts **Positive, Neutral, or Negative** sentiment.

## Pipeline

`Product Review -> Preprocessing -> Aspect Detection -> Aspect Context Extraction -> BiLSTM -> Sentiment`

## Project files

- `generate_dataset.py` — creates a balanced teaching dataset.
- `train_model.py` — trains and evaluates the BiLSTM model.
- `aspect_utils.py` — text cleaning, keyword-based aspect detection, and simple clause/context extraction.
- `app.py` — Streamlit web interface.
- `data/reviews.csv` — generated training data.
- `model/sentiment_model.keras` — saved model after training.
- `model/metrics.json` — test metrics after training.

## Setup

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\\Scripts\\activate     # Windows
pip install -r requirements.txt
```

## Generate data and train

```bash
python generate_dataset.py
python train_model.py
```

## Run the app

```bash
streamlit run app.py
```

## Example

Input:

> The camera takes amazing pictures and the display is beautiful, but the battery drains quickly. The price is too high.

Expected interpretation:

- Camera -> Positive
- Display -> Positive
- Battery -> Negative
- Price -> Negative

## Viva explanation

1. The review is cleaned.
2. Predefined keywords detect supported aspects.
3. The review is split into simple clauses so each aspect gets its relevant context.
4. The context is converted into integer tokens by Keras `TextVectorization`.
5. An Embedding layer learns vector representations of words.
6. A Bidirectional LSTM reads the sequence in both directions and learns sentiment patterns.
7. A Softmax output layer returns probabilities for Negative, Neutral, and Positive sentiment.

## Important limitation

This project intentionally uses a **synthetic teaching dataset** so it is easy to run and understand in an NLP lab. It demonstrates the complete deep-learning pipeline, but it should not be presented as a production model. For a research-quality system, replace `data/reviews.csv` with a large human-annotated ABSA/product-review dataset and retrain.
