
import os
import argparse
import json
import joblib
import re
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def clean_text(s):
    s = str(s)
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"\S+@\S+", " ", s)
    s = re.sub(r"[^0-9A-Za-z'\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def plot_confusion(cm, labels, outpath):
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt='d', cbar=False, xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def main(args):
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    df = pd.read_csv(args.data)
    df = df.dropna(subset=[args.text_col, args.label_col])
    df["text"] = df[args.text_col].astype(str).map(clean_text)
    y = df[args.label_col].astype(str)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(
        df["text"], y_enc, test_size=args.val_split, stratify=y_enc, random_state=args.random_state
    )

    vectorizer = TfidfVectorizer(
    ngram_range=(1,2),      # include unigrams + bigrams
    max_features=10000,     # raise cap if memory allows
    stop_words='english'    # remove common stop words
)

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    models = {
        "logistic": LogisticRegression(max_iter=1000, class_weight="balanced", solver="saga", n_jobs=-1, random_state=args.random_state),
        "randomforest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=args.random_state,class_weight="balanced"),
        "naive_bayes": MultinomialNB()
    }

    results = {}
    best_model = None
    best_f1 = -1
    for name, m in models.items():
        print(f"\nTraining {name} ...")
        m.fit(X_train_tfidf, y_train)
        preds = m.predict(X_val_tfidf)
        acc = accuracy_score(y_val, preds)
        f1w = f1_score(y_val, preds, average="weighted")
        print(f"{name}  ACC={acc:.4f}  F1w={f1w:.4f}")
        print(classification_report(y_val, preds, target_names=le.classes_))
        results[name] = {"acc": acc, "f1_weighted": f1w}
        if f1w > best_f1:
            best_f1 = f1w
            best_model = m
            best_name = name

    # Save artifacts
    joblib.dump(vectorizer, "models/vectorizer.pkl")
    joblib.dump(best_model, "models/model.pkl")
    joblib.dump(le, "models/label_encoder.pkl")

    # Confusion matrix for best model
    preds_best = best_model.predict(X_val_tfidf)
    cm = confusion_matrix(y_val, preds_best)
    plot_confusion(cm, list(le.classes_), "reports/confusion_matrix.png")

    # Save metrics
    with open("reports/metrics.json", "w") as f:
        json.dump({"results": results, "best_model": best_name}, f, indent=2)

    print(f"\nSaved best model: {best_name} (F1={best_f1:.4f}) -> models/model.pkl")
    print("Artifacts: models/* and reports/*")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset.csv")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--max-features", type=int, default=100000)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()
    main(args)
