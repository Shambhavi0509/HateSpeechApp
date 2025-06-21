import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords

# 📥 Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ⚠️ Load trained model and vectorizer
model = joblib.load("hate_speech_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# 🎯 Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# 🌐 Streamlit page setup
st.set_page_config(page_title="Hate Speech Detector", page_icon="⚠️")

# 🧢 Custom Title
st.markdown("""
    <style>
    .title {
        font-size:36px;
        font-weight:bold;
        color:#FF4B4B;
        text-align:center;
    }
    .subtext {
        font-size:18px;
        color:gray;
        text-align:center;
    }
    </style>
    <div class='title'>🚫 Hate Speech Detection App</div>
    <div class='subtext'>Built with ML • NLP • Streamlit</div>
    <hr>
""", unsafe_allow_html=True)

# 📘 Sidebar info
st.sidebar.title("About This App")
st.sidebar.markdown("""
This app detects **hate or offensive content** using a trained ML model.

- 🧠 Algorithm: Logistic Regression
- ✨ NLP: Tfidf + Stopword Removal
- ⚙️ Framework: Streamlit

👩‍💻 Created by: *Your Name*
""")

# 💡 Example inputs
examples = {
    "Example 1: Positive": "I love everyone",
    "Example 2: Offensive": "You're an idiot",
    "Example 3: Aggressive": "They deserve to die",
    "Example 4: Kind": "You are amazing",
    "Example 5: Neutral": "Let's go for a walk",
}

selected = st.selectbox("💡 Try a sample input:", list(examples.keys()))
text_input = st.text_area("📝 Or type your own text below:", value=examples[selected])

# 🧠 Predict
if st.button("Detect"):
    cleaned = clean_text(text_input)
    vect = vectorizer.transform([cleaned])
    prob = model.predict_proba(vect)[0][1]

    # 🎯 Show result
    if prob > 0.5:
        st.error("⚠️ Hate or Offensive Speech Detected")
    else:
        st.success("✅ This text is safe.")

    # 📊 Show confidence
    st.progress(min(prob, 1.0))
    st.write(f"🧠 Confidence: {round(prob * 100, 2)}%")

# 🔬 Expandable Model Info
with st.expander("🔬 See Model Details"):
    st.markdown("""
    - **Model:** Logistic Regression (balanced)
    - **Vectorizer:** TfidfVectorizer
    - **Text Cleaning:** Lowercase, stopwords, punctuation removal
    - **Dataset:** 10-line sample dataset
    """)

# 📬 Feedback input
st.text_input("💬 Want to share feedback or suggestions?")
