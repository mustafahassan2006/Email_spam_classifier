import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords inside the app
@st.cache_resource
def load_assets():
    nltk.download('stopwords', quiet=True)
    model = joblib.load('spam_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    stop_words = set(stopwords.words('english'))
    return model, tfidf, stop_words

model, tfidf, stop_words = load_assets()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()
    return " ".join([w for w in words if w not in stop_words])

st.title("Email Spam Classifier")
st.write("Enter an email message below to check whether it's Spam or Ham.")

user_input = st.text_area("Email Content", height=150)

if st.button("Classify"):
    if user_input.strip() != "":
        cleaned = clean_text(user_input)
        vec = tfidf.transform([cleaned])
        pred = model.predict(vec)[0]
        
        if pred == 1:
            st.error("🚨  SPAM!")
        else:
            st.success("✅ NOT SPAM.")
    else:
        st.warning("Please enter some text first.")