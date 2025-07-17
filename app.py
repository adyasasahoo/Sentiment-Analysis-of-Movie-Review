import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("🎬 Movie Review Sentiment Analyzer")

review = st.text_area("Enter a movie review:")

if st.button("Analyze"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        vec = vectorizer.transform([review])
        prediction = model.predict(vec)[0]
        if prediction == 1:
            st.success("😊 Positive Sentiment")
        else:
            st.error("😞 Negative Sentiment")