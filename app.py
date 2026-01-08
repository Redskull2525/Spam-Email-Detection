import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ----------------------------------
# Page configuration
# ----------------------------------
st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧",
    layout="centered"
)

# ----------------------------------
# Load model and tokenizer
# ----------------------------------
@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model("model/spam_model.keras")
    with open("model/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Same max_len used during training
MAX_LEN = 100

# ----------------------------------
# UI
# ----------------------------------
st.title("📧 Spam Email Detection")
st.write(
    "This app uses a **Deep Learning LSTM model** to detect whether an email is **Spam** or **Not Spam**."
)

email_text = st.text_area(
    "✉️ Enter Email Text",
    placeholder="Congratulations! You have won a free laptop. Click here to claim."
)

# ----------------------------------
# Prediction
# ----------------------------------
if st.button("🔍 Predict"):
    if email_text.strip() == "":
        st.warning("⚠️ Please enter some email text.")
    else:
        # Preprocessing
        sequence = tokenizer.texts_to_sequences([email_text])
        padded_sequence = pad_sequences(
            sequence, maxlen=MAX_LEN, padding="post", truncating="post"
        )

        # Prediction
        prediction = model.predict(padded_sequence)[0][0]

        st.subheader("📊 Prediction Result")

        if prediction > 0.5:
            st.error(f"🚨 Spam Email\n\nConfidence: {prediction:.2f}")
        else:
            st.success(f"✅ Not Spam\n\nConfidence: {1 - prediction:.2f}")

# ----------------------------------
# Footer
# ----------------------------------
st.markdown("---")
st.caption("Built with ❤️ using TensorFlow & Streamlit")
