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

# 2. Custom CSS to match website theme
st.markdown("""
    <style>
    .stApp {
        background-color: #0a192f;
        color: #ccd6f6;
        font-family: 'Inter', sans-serif;
    }

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

    .warning-card {
        background-color: rgba(255, 193, 7, 0.1);
        border: 1px solid #ffc107;
        color: #ffc107;
        box-shadow: 0 0 15px rgba(255, 193, 7, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

# 3. Navigation Header
PORTFOLIO_URL = "https://mustafahassan.site" 
st.markdown(f'<a href="{PORTFOLIO_URL}" class="portfolio-link">← Home</a>', unsafe_allow_html=True)

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

# Initialize Session State
if "show_result_modal" not in st.session_state:
    st.session_state.show_result_modal = False
if "modal_data" not in st.session_state:
    st.session_state.modal_data = {}

# Check if user clicked close link FIRST
if st.query_params.get("close") == "true":
    st.session_state.show_result_modal = False
    st.query_params.clear()

user_input = st.text_area("Email Content", height=120, placeholder="Paste your email text here...")

# Analyze Button Logic
if st.button("Analyze & Classify"):
    st.query_params.clear()
    
    if user_input.strip() != "":
        cleaned = clean_text(user_input)
        vec = tfidf.transform([cleaned])
        pred = model.predict(vec)[0]
        
        if pred == 1:
            st.session_state.modal_data = {
                "title": "🚨 SPAM DETECTED",
                "card_class": "spam-card",
                "desc": "This message contains patterns commonly associated with phishing or scam emails."
            }
        else:
            st.session_state.modal_data = {
                "title": "✅ NOT SPAM (HAM)",
                "card_class": "ham-card",
                "desc": "This message appears safe and resembles normal communication."
            }
    else:
        st.session_state.modal_data = {
            "title": "⚠️ INPUT REQUIRED",
            "card_class": "warning-card",
            "desc": "Please enter or paste some email text before running the classification."
        }
    
    st.session_state.show_result_modal = True

# Render Modal Overlay
if st.session_state.show_result_modal:
    data = st.session_state.modal_data
    
    st.markdown(f"""
        <style>
        .stApp {{
            overflow: hidden !important;
        }}
        .modal-overlay {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(10, 25, 47, 0.90);
            backdrop-filter: blur(6px);
            z-index: 999998;
        }}
        .modal-card-container {{
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: #112240;
            border: 1px solid #64ffda;
            padding: 30px 25px;
            border-radius: 16px;
            text-align: center;
            max-width: 400px;
            width: 85%;
            box-shadow: 0 20px 40px rgba(0,0,0,0.6);
            z-index: 999999;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .close-link-btn {{
            display: inline-block;
            background: transparent;
            color: #64ffda !important;
            border: 1.5px solid #64ffda;
            border-radius: 50px;
            padding: 8px 30px;
            font-size: 0.95rem;
            font-weight: 600;
            text-decoration: none !important;
            margin-top: 20px;
            transition: all 0.3s ease;
        }}
        .close-link-btn:hover {{
            background: rgba(100, 255, 218, 0.1);
            box-shadow: 0 0 15px rgba(100, 255, 218, 0.3);
        }}
        </style>
        
        <div class="modal-overlay"></div>
        <div class="modal-card-container">
            <div class="result-card {data['card_class']}" style="margin: 0; width: 100%;">{data['title']}</div>
            <p style="color: #8892b0; margin: 18px 0 0 0; font-size: 0.95rem; line-height: 1.5;">{data['desc']}</p>
            <a href="?close=true" target="_self" class="close-link-btn">✕ Close</a>
        </div>
    """, unsafe_allow_html=True)

    #
    
