import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# 1. Page Configuration
st.set_page_config(
    page_title="Spam Classifier | Mustafa Hassan",
    page_icon="🛡️",
    layout="centered"
)

# 2. Custom CSS to match website theme (#0a192f, #64ffda, #ccd6f6)
st.markdown("""
    <style>
    /* Dark Navy Background */
    .stApp {
        background-color: #0a192f;
        color: #ccd6f6;
        font-family: 'Inter', sans-serif;
    }

    /* Top Portfolio Link */
    .portfolio-link {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        color: #64ffda;
        text-decoration: none;
        font-family: monospace;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
        padding: 6px 14px;
        border-radius: 20px;
        background: rgba(100, 255, 218, 0.05);
        border: 1px solid rgba(100, 255, 218, 0.2);
        transition: all 0.3s ease;
    }

    .portfolio-link:hover {
        background: rgba(100, 255, 218, 0.15);
        border-color: #64ffda;
        box-shadow: 0 0 10px rgba(100, 255, 218, 0.3);
    }
    
    /* Title and Headers */
    h1 {
        color: #f8fafc !important;
        font-weight: 800 !important;
        text-align: center;
        margin-bottom: 0px !important;
    }
    
    .sub-title {
        color: #64ffda;
        text-align: center;
        font-family: monospace;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    .description {
        color: #8892b0;
        text-align: center;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    /* Custom Textarea */
    .stTextArea textarea {
        background-color: #112240 !important;
        color: #ccd6f6 !important;
        border: 1px solid #233554 !important;
        border-radius: 8px !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #64ffda !important;
        box-shadow: 0 0 10px rgba(100, 255, 218, 0.2) !important;
    }

    /* Modern Glowing Button */
    .stButton > button {
        width: 100%;
        background-color: transparent !important;
        color: #64ffda !important;
        border: 1.5px solid #64ffda !important;
        padding: 12px 24px !important;
        border-radius: 50px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        background-color: rgba(100, 255, 218, 0.1) !important;
        box-shadow: 0 0 15px rgba(100, 255, 218, 0.3) !important;
        transform: translateY(-2px);
    }

    /* Custom Result Cards */
    .result-card {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 20px;
    }
    
    .spam-card {
        background-color: rgba(255, 99, 132, 0.1);
        border: 1px solid #ff6384;
        color: #ff6384;
    }

    .ham-card {
        background-color: rgba(100, 255, 218, 0.1);
        border: 1px solid #64ffda;
        color: #64ffda;
        box-shadow: 0 0 15px rgba(100, 255, 218, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

# 3. Navigation Header (Update URL with your live domain)
PORTFOLIO_URL = "https://mustafahassan.site" 
st.markdown(f'<a href="{PORTFOLIO_URL}" class="portfolio-link"> Home</a>', unsafe_allow_html=True)

# 4. Model & NLTK Loading
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

# 5. Interface Layout
st.markdown("<h1>Email Spam Classifier</h1>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'><span>01.</span> Natural Language Processing</div>", unsafe_allow_html=True)
st.markdown("<p class='description'>Enter an email message below to analyze text patterns and classify whether it's <b>Spam</b> or <b>Ham</b>.</p>", unsafe_allow_html=True)

user_input = st.text_area("Email Content", height=150, placeholder="Paste your email text here...")

if st.button("Analyze & Classify"):
    if user_input.strip() != "":
        cleaned = clean_text(user_input)
        vec = tfidf.transform([cleaned])
        pred = model.predict(vec)[0]
        
        if pred == 1:
            st.markdown("<div class='result-card spam-card'>🚨 SPAM DETECTED</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-card ham-card'>✅ NOT SPAM (HAM)</div>", unsafe_allow_html=True)
    else:
        st.warning("Please enter some email content first.")