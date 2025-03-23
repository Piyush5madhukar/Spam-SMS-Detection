import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained LSTM model
model = load_model("spam_classifier_lstm.h5")

# Load the tokenizer
with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

def classify_sms(text):
    """Predict if an SMS is spam or ham."""
    seq = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(seq, maxlen=50)
    prediction = (model.predict(padded_seq) > 0.5).astype("int32")
    return "Spam" if prediction[0][0] == 1 else "Ham"

# Streamlit UI
st.title("📩 SMS Spam Detector")
st.write("Enter an SMS message below to check if it's Spam or Ham.")

# Text input
sms_text = st.text_area("Enter SMS Message:", "")

if st.button("Classify SMS"):
    if sms_text.strip():
        result = classify_sms(sms_text)
        if result == "Spam":
            st.error("🚨 This message is Spam!")
        else:
            st.success("✅ This message is Ham (Not Spam).")
    else:
        st.warning("Please enter a message to classify.")

# Run using: streamlit run filename.py
