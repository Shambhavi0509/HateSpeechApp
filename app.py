import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
model = joblib.load("hate_speech_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# UI
st.set_page_config(page_title="Hate Speech Detector", page_icon="⚠️")
st.title("🚫 Hate Speech Detection App")
st.write("Enter a sentence to check if it contains hate or offensive content.")

text_input = st.text_area("📝 Type your text here:")

if st.button("Detect"):
    cleaned = clean_text(text_input)
    vect = vectorizer.transform([cleaned])
    prob = model.predict_proba(vect)[0][1]
    if prob > 0.5:
        st.error(f"⚠️ Hate or Offensive Speech Detected (Confidence: {round(prob*100, 2)}%)")
    else:
        st.success(f"✅ This text is safe. (Confidence: {round((1 - prob)*100, 2)}%)")
