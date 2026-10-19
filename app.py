import streamlit as st
import tensorflow as tf
import pickle
import os
import string
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧",
    layout="centered"
)

# -----------------------------
# Constants
# -----------------------------
MODEL_PATH = "model/spam_model.keras"
TOKENIZER_PATH = "model/tokenizer.pkl"
STOPWORDS_PATH = "model/stopwords.txt"
MAX_LEN = 100
THRESHOLD = 0.65

# -----------------------------
# Load model & tokenizer safely
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
        return None, None

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer


model, tokenizer = load_model_and_tokenizer()


@st.cache_resource(show_spinner=False)
def load_stopwords():
    if not os.path.exists(STOPWORDS_PATH):
        return set()
    with open(STOPWORDS_PATH, encoding="utf-8") as f:
        return set(f.read().split())


STOP_WORDS = load_stopwords()
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def clean_text(text):
    """Same cleaning as training.ipynb: drop 'Subject', punctuation, stopwords; lowercase."""
    text = text.replace("Subject", "").translate(PUNCT_TABLE)
    return " ".join(w.lower() for w in text.split() if w.lower() not in STOP_WORDS)

# -----------------------------
# UI
# -----------------------------
st.title("📧 Spam Email Detection")
st.write(
    "This app uses a **Deep Learning LSTM model** to classify emails as "
    "**Spam** or **Not Spam**."
)

# -----------------------------
# Handle missing model case
# -----------------------------
if model is None or tokenizer is None:
    st.warning("⚠️ Model files not found.")

    st.markdown(
        """
        ### How to fix this:
        1. Run `notebooks/training.ipynb` from the repo root to train the model  
        2. This will create:
           - `model/spam_model.keras`
           - `model/tokenizer.pkl`
        3. Then run:
           ```
           streamlit run app.py
           ```
        """
    )
    st.stop()

# -----------------------------
# Input
# -----------------------------
email_text = st.text_area(
    "✉️ Enter Email Text",
    placeholder="You have won a free laptop! Click here to claim."
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict"):
    if email_text.strip() == "":
        st.warning("⚠️ Please enter some email text.")
    else:
        sequence = tokenizer.texts_to_sequences([clean_text(email_text)])
        padded = pad_sequences(sequence, maxlen=MAX_LEN, padding="post")

        probability = model.predict(padded, verbose=0)[0][0]

        st.subheader("📊 Prediction Result")

        if probability >= THRESHOLD:
            st.error(f"🚨 Spam Email\n\nConfidence: {probability:.2f}")
        else:
            st.success(f"✅ Not Spam\n\nConfidence: {1 - probability:.2f}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Built with ❤️ using TensorFlow & Streamlit")
