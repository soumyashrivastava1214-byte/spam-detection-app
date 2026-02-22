import streamlit as st
import joblib
import re

# Load saved files
model = joblib.load("spam_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
le = joblib.load("label_encoder.pkl")

# Text cleaning function (same as training)
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# UI
st.set_page_config(page_title="Spam Detector", layout="centered")
st.title("📧 Spam Detection App")
st.write("Enter a message below to check whether it is **Spam** or **Ham**.")

user_input = st.text_area("Message", height=150)

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        cleaned = clean_text(user_input)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0][prediction]

        label = le.inverse_transform([prediction])[0]

        if label == "spam":
            st.error(f"🚨 SPAM detected (confidence: {probability:.2f})")
        else:
            st.success(f"✅ HAM (not spam) (confidence: {probability:.2f})")