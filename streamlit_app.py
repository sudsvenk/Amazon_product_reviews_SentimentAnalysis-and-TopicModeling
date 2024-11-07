import streamlit as st
#import joblib
import pickle
#lda_model = joblib.load('lda_model.joblib')
#vectorizer = joblib.load('vectorizer.joblib')
# Load the saved model and vectorizer
with open("lda_model.pkl", "rb") as file:
    lda_model = pickle.load(file)
with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# Define preprocess_text here or import if defined elsewhere
def preprocess_text(text):
    # Include preprocessing steps here
    return text

st.title("Amazon Reviews Topic Model")
user_input = st.text_area("Enter a review to find its topic:")

if st.button("Predict Topic"):
    processed_text = preprocess_text(user_input)
    text_vectorized = vectorizer.transform([processed_text])
    topic_dist = lda_model.transform(text_vectorized)
    dominant_topic = topic_dist.argmax()
    st.write(f"Predicted Topic: Topic #{dominant_topic + 1}")
