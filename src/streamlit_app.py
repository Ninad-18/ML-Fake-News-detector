
import streamlit as st
import joblib

@st.cache_resource
def load():
    model = joblib.load("models/model.pkl")
    vec = joblib.load("models/vectorizer.pkl")
    le = joblib.load("models/label_encoder.pkl")
    return model, vec, le

st.title("Fake News Classifier — TF-IDF + baseline")
st.write("Paste an article/headline and press Predict")

model, vec, le = load()

text = st.text_area("News text", height=200)
if st.button("Predict"):
    if not text.strip():
        st.warning("Paste some text first.")
    else:
        X = vec.transform([text])
        pred = model.predict(X)[0]
        label = le.inverse_transform([pred])[0]
        conf = None
        if hasattr(model, "predict_proba"):
            conf = float(model.predict_proba(X)[0].max())
        st.subheader(f"Prediction: {label.upper()}")
        if conf is not None:
            st.write(f"Confidence: {conf:.2f}")
