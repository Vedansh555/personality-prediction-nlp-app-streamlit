# Paste your full Streamlit code here 👇

import streamlit as st
import joblib
import matplotlib.pyplot as plt

# Load model + vectorizer
import pickle
import requests

def download_and_load(url, local_filename):
    r = requests.get(url)
    with open(local_filename, 'wb') as f:
        f.write(r.content)
    with open(local_filename, 'rb') as f:
        return pickle.load(f)

# 🔗 Paste your real direct links here

model_url = 'https://drive.google.com/uc?export=download&id=1Cn7HmkkF2oZzZbJYq7e6xsLd4g3RSrKW'
tfidf_url = 'https://drive.google.com/uc?export=1PZzSj2EJGIc3X9SffcY27puBERwpuL-B'
model = download_and_load(model_url, 'multi_rf.pkl')
vectorizer = download_and_load(tfidf_url, 'tfidf.pkl')


traits = ['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']

def plot_traits(pred):
    colors = ['green' if v == 1 else 'red' for v in pred]
    plt.barh(traits, pred, color=colors)
    plt.xlim(0, 1.2)
    plt.xlabel("Prediction")
    plt.title("Personality Traits")
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    st.pyplot(plt)

st.title("🧠 Personality Predictor")
text = st.text_area("Paste your text:")

if st.button("Predict"):
    if text.strip():
        vec = tfidf.transform([text])
        pred = model.predict(vec)[0].tolist()
        st.subheader("Prediction Results:")
        for trait, val in zip(traits, pred):
            st.write(f"**{trait}**: {'✅ Present' if val else '❌ Absent'}")
        plot_traits(pred)
    else:
        st.warning("Please enter some text.")
