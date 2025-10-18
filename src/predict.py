
import joblib
import argparse
import numpy as np

def load_artifacts(model_p="models/model.pkl", vec_p="models/vectorizer.pkl", le_p="models/label_encoder.pkl"):
    model = joblib.load(model_p)
    vec = joblib.load(vec_p)
    le = joblib.load(le_p)
    return model, vec, le

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", nargs="+", help="One or more texts", required=True)
    args = ap.parse_args()
    model, vec, le = load_artifacts()
    texts = [" ".join(args.text)]
    X = vec.transform(texts)
    preds = model.predict(X)
    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
    labels = le.inverse_transform(preds)
    for i, t in enumerate(texts):
        p = probs[i].max() if probs is not None else None
        print(f"LABEL={labels[i]}  CONFIDENCE={p}  TEXT={t[:120]}...")
