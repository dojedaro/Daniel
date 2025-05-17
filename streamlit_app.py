import streamlit as st
import requests

# Load Hugging Face token from Streamlit Cloud secrets
HF_TOKEN = st.secrets["HF_TOKEN"]

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# ✅ Working model that supports Korean ↔ English
API_URL = "https://api-inference.huggingface.co/models/facebook/nllb-200-distilled-600M"

# Language codes for NLLB model
LANG_CODES = {
    "English → Korean": ("eng_Latn", "kor_Hang"),
    "Korean → English": ("kor_Hang", "eng_Latn")
}

# Translation function
def translate(text, direction):
    src_lang, tgt_lang = LANG_CODES[direction]
    payload = {
        "inputs": text,
        "parameters": {
            "src_lang": src_lang,
            "tgt_lang": tgt_lang
        }
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        try:
            return response.json()[0]["translation_text"]
        except Exception as e:
            return f"⚠️ Translation succeeded but response format was unexpected: {e}"
    else:
        return f"❌ Error: {response.status_code} — Try again later."

# Streamlit App UI
st.set_page_config(page_title="English ↔ Korean Translator")
st.title("English ↔ Korean Translator")

direction = st.radio("Select translation direction", ["English → Korean", "Korean → English"])
text = st.text_area("Enter your text:")

if st.button("Translate"):
    if text.strip():
        result = translate(text, direction)
        st.success(result)
