import streamlit as st
import requests

HF_TOKEN = st.secrets["HF_TOKEN"]

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

API_URLS = {
    "English → Korean": "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-ko",
    "Korean → English": "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-ko-en",
}

def query(payload, direction):
    response = requests.post(API_URLS[direction], headers=headers, json={"inputs": payload})
    if response.status_code == 200:
        return response.json()[0]["translation_text"]
    else:
        return f"Error: {response.status_code} — Try again later."
