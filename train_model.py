import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE = Path(__file__).parent
DATA = BASE / "data" / "reviews.csv"
MODEL_DIR = BASE / "model"
MODEL_DIR.mkdir(exist_ok=True)

np.random.seed(42)

try:
    import tensorflow as tf

    tf.random.set_seed(42)

    df = pd.read_csv(DATA)
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    y = df["sentiment"].map(label_map).values
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].astype(str).values, y, test_size=0.20, random_state=42, stratify=y
    )

    max_tokens = 5000
    seq_len = 30
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=seq_len,
        standardize="lower_and_strip_punctuation",
    )
    vectorizer.adapt(X_train)

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(1,), dtype=tf.string),
        vectorizer,
        tf.keras.layers.Embedding(max_tokens, 64, mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, validation_split=0.15, epochs=8, batch_size=32, verbose=1)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    pred = model.predict(X_test, verbose=0).argmax(axis=1)
    report = classification_report(y_test, pred, target_names=["negative", "neutral", "positive"], output_dict=True)
    cm = confusion_matrix(y_test, pred).tolist()

    model.save(MODEL_DIR / "sentiment_model.keras")
    metrics = {"test_accuracy": float(acc), "classification_report": report, "confusion_matrix": cm, "labels": label_map}
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test, pred, target_names=["negative", "neutral", "positive"]))
except Exception as exc:
    print(f"TensorFlow unavailable ({exc}); training a scikit-learn fallback instead.")

    df = pd.read_csv(DATA)
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    y = df["sentiment"].map(label_map).values
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].astype(str).values, y, test_size=0.20, random_state=42, stratify=y
    )

    model = Pipeline([
        ("vectorizer", CountVectorizer(ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42, multi_class="auto")),
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    report = classification_report(y_test, pred, target_names=["negative", "neutral", "positive"], output_dict=True)
    cm = confusion_matrix(y_test, pred).tolist()
    acc = float(np.mean(pred == y_test))

    import joblib

    joblib.dump(model, MODEL_DIR / "sentiment_model.joblib")
    metrics = {"test_accuracy": float(acc), "classification_report": report, "confusion_matrix": cm, "labels": label_map}
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Fallback accuracy: {acc:.4f}")
    print(classification_report(y_test, pred, target_names=["negative", "neutral", "positive"]))
