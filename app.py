from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from aspect_utils import clean_text, extract_aspect_contexts

BASE = Path(__file__).parent
MODEL_PATH = BASE / "model" / "sentiment_model.keras"
FALLBACK_MODEL_PATH = BASE / "model" / "sentiment_model.joblib"
LABELS = ["Negative", "Neutral", "Positive"]
EMOJI = {"Positive": "🟢", "Neutral": "🟡", "Negative": "🔴"}

st.set_page_config(page_title="AspectSense", layout="centered")

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        import tensorflow as tf

        return ("tensorflow", tf.keras.models.load_model(MODEL_PATH))
    if FALLBACK_MODEL_PATH.exists():
        return ("sklearn", joblib.load(FALLBACK_MODEL_PATH))
    return ("missing", None)

def predict(model, text):
    kind, estimator = model
    if kind == "tensorflow":
        probs = estimator.predict(np.array([text]), verbose=0)[0]
        idx = int(np.argmax(probs))
        return LABELS[idx], float(probs[idx])

    probs = estimator.predict_proba([text])[0]
    idx = int(np.argmax(probs))
    return LABELS[idx], float(probs[idx])

st.title("AspectSense")

if not (MODEL_PATH.exists() or FALLBACK_MODEL_PATH.exists()):
    st.error("Model not found. Run `python generate_dataset.py` and then `python train_model.py` first.")
    st.stop()

model = load_model()
model_name = "BiLSTM" if model[0] == "tensorflow" else "Logistic Regression"
st.caption(f"Explore product-review sentiment by aspect · Model: {model_name}")

st.session_state.setdefault(
    "review_input",
    "The camera takes amazing pictures and the display is beautiful, but the battery drains quickly. The price is too high.",
)
with st.form("review_form"):
    review = st.text_area(
        "Enter an electronic product review",
        key="review_input",
        height=140,
        placeholder="Mention camera, battery, display/screen, or price/cost...",
    )
    submitted = st.form_submit_button("Analyze review", type="primary", use_container_width=True)

if submitted:
    if not review.strip():
        st.session_state.pop("analysis", None)
        st.warning("Please enter a review.")
    else:
        contexts = extract_aspect_contexts(review)
        st.session_state["analysis"] = {
            "review": review,
            "contexts": contexts,
            "predictions": {
                aspect: predict(model, context)
                for aspect, context in contexts.items()
            },
        }

analysis = st.session_state.get("analysis")
if analysis:
    if analysis["review"] != st.session_state["review_input"]:
        st.caption("Showing results for the last submitted review. Analyze again to update.")

    contexts = analysis["contexts"]
    predictions = analysis["predictions"]
    if not contexts:
        st.info("No supported aspect was detected. Try mentioning camera, battery, display/screen, or price/cost.")
    else:
        st.subheader("Aspect-wise results")
        options = list(contexts)
        selected_aspects = st.multiselect(
            "Filter results by aspect",
            options=options,
            default=options,
        )
        visible = [aspect for aspect in options if aspect in selected_aspects]
        positive_count = sum(predictions[aspect][0] == "Positive" for aspect in visible)
        negative_count = sum(predictions[aspect][0] == "Negative" for aspect in visible)
        neutral_count = sum(predictions[aspect][0] == "Neutral" for aspect in visible)

        total_col, positive_col, neutral_col, negative_col = st.columns(4)
        total_col.metric("Aspects shown", len(visible))
        positive_col.metric("Positive", positive_count)
        neutral_col.metric("Neutral", neutral_count)
        negative_col.metric("Negative", negative_count)

        for aspect in visible:
            sentiment, confidence = predictions[aspect]
            with st.container(border=True):
                result_col, confidence_col = st.columns([3, 2])
                result_col.markdown(f"### {EMOJI[sentiment]} {aspect}")
                result_col.write(f"Sentiment: **{sentiment}**")
                confidence_col.write(f"Model confidence: **{confidence * 100:.1f}%**")
                confidence_col.progress(confidence)
                st.caption(f"Context: {contexts[aspect]}")

        download_rows = [
            {
                "Aspect": aspect,
                "Sentiment": predictions[aspect][0],
                "Confidence": f"{predictions[aspect][1] * 100:.1f}%",
                "Context": contexts[aspect],
            }
            for aspect in visible
        ]
        st.download_button(
            "Download visible results as CSV",
            data=pd.DataFrame(download_rows).to_csv(index=False),
            file_name="aspect-sentiment-results.csv",
            mime="text/csv",
        )

        with st.expander("Explore the NLP pipeline"):
            st.write("**Cleaned review**")
            st.code(clean_text(analysis["review"]))
            st.write("**Detected aspects and extracted context**")
            for aspect in visible:
                st.write(f"**{aspect}:** {contexts[aspect]}")
            st.write("**Sentiment model**")
            st.write(f"The {model_name} model classifies each aspect context as Positive, Neutral, or Negative.")

with st.expander("How it works"):
    st.write("Review text is cleaned, split into clauses, and matched to supported aspect keywords before sentiment is predicted.")
    st.caption("Supported aspects: Camera, Battery, Display, Price")
