import os
import argparse
import json
import joblib
import re
import pandas as pd
import numpy as np
from collections import Counter

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

import matplotlib.pyplot as plt
import seaborn as sns

class LinguisticFeatures(BaseEstimator, TransformerMixin):
    """Extract linguistic features from text"""
    
    def __init__(self):
        # Simple stop words list
        self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
    
    def extract_features(self, text):
        """Extract various linguistic features"""
        features = {}
        
        # Basic text statistics
        words = text.lower().split()
        sentences = text.split('.')
        
        features['char_count'] = len(text)
        features['word_count'] = len(words)
        features['sentence_count'] = len(sentences)
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        features['avg_sentence_length'] = np.mean([len(sent.split()) for sent in sentences]) if sentences else 0
        
        # Punctuation and special characters
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        # Word-level features
        features['stopword_ratio'] = sum(1 for word in words if word in self.stop_words) / len(words) if words else 0
        features['unique_word_ratio'] = len(set(words)) / len(words) if words else 0
        
        # Simple sentiment indicators
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'positive', 'success', 'win', 'victory', 'best', 'better', 'improve', 'increase', 'rise', 'gain'}
        negative_words = {'bad', 'terrible', 'awful', 'horrible', 'negative', 'fail', 'loss', 'defeat', 'crisis', 'problem', 'worst', 'worse', 'decline', 'decrease', 'fall', 'drop'}
        
        features['positive_word_ratio'] = sum(1 for word in words if word in positive_words) / len(words) if words else 0
        features['negative_word_ratio'] = sum(1 for word in words if word in negative_words) / len(words) if words else 0
        
        # Fake news indicators
        fake_indicators = {'breaking', 'shocking', 'unbelievable', 'exclusive', 'urgent', 'alert', 'warning', 'conspiracy', 'cover-up', 'scandal', 'exposed', 'revealed', 'leaked'}
        features['fake_indicator_ratio'] = sum(1 for word in words if word in fake_indicators) / len(words) if words else 0
        
        # Numbers and statistics
        features['number_count'] = len(re.findall(r'\d+', text))
        features['number_ratio'] = features['number_count'] / len(words) if words else 0
        
        # URL and email indicators
        features['url_count'] = len(re.findall(r'http\S+|www\.\S+', text))
        features['email_count'] = len(re.findall(r'\S+@\S+', text))
        
        return features
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features_list = []
        for text in X:
            features = self.extract_features(text)
            features_list.append(list(features.values()))
        return np.array(features_list)

def clean_text_advanced(text):
    """Advanced text cleaning"""
    text = str(text)
    
    # Remove URLs and emails
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    
    # Remove extra whitespace and normalize
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def plot_confusion_matrix(cm, labels, outpath):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cbar=True, xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()

def plot_feature_importance(model, feature_names, outpath, top_n=20):
    """Plot feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 8))
        plt.title(f"Top {top_n} Feature Importances")
        plt.bar(range(top_n), importances[indices])
        plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close()

def main(args):
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    print("Loading and preprocessing data...")
    df = pd.read_csv(args.data)
    df = df.dropna(subset=[args.text_col, args.label_col])
    
    # Advanced text cleaning
    df["text"] = df[args.text_col].astype(str).map(clean_text_advanced)
    y = df[args.label_col].astype(str)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution:\n{y.value_counts()}")
    
    # Handle class imbalance
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    # Stratified split to maintain class distribution
    X_train, X_val, y_train, y_val = train_test_split(
        df["text"], y_enc, test_size=args.val_split, stratify=y_enc, random_state=args.random_state
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    
    # Enhanced TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),  # Include trigrams
        max_features=args.max_features,
        stop_words='english',
        min_df=2,  # Ignore terms that appear in less than 2 documents
        max_df=0.95,  # Ignore terms that appear in more than 95% of documents
        sublinear_tf=True,  # Apply sublinear tf scaling
        norm='l2'  # L2 normalization
    )
    
    # Linguistic features extractor
    linguistic_extractor = LinguisticFeatures()
    
    # Transform training data
    print("Extracting features...")
    
    # Convert Series to DataFrame for ColumnTransformer
    X_train_df = pd.DataFrame({'text': X_train})
    X_val_df = pd.DataFrame({'text': X_val})
    
    # Create feature pipeline
    feature_pipeline = ColumnTransformer([
        ('tfidf', tfidf_vectorizer, 'text'),
        ('linguistic', linguistic_extractor, 'text')
    ])
    
    X_train_features = feature_pipeline.fit_transform(X_train_df)
    X_val_features = feature_pipeline.transform(X_val_df)
    
    print(f"Feature matrix shape: {X_train_features.shape}")
    
    # Enhanced models with better hyperparameters
    models = {
        "logistic": LogisticRegression(
            max_iter=2000, 
            class_weight="balanced", 
            solver="saga", 
            n_jobs=-1, 
            random_state=args.random_state,
            C=0.1  # Regularization
        ),
        "randomforest": RandomForestClassifier(
            n_estimators=300, 
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1, 
            random_state=args.random_state,
            class_weight="balanced"
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=args.random_state
        ),
        "svm": SVC(
            kernel='rbf',
            C=1.0,
            class_weight="balanced",
            probability=True,
            random_state=args.random_state
        ),
        "naive_bayes": MultinomialNB(alpha=0.1)
    }
    
    # Ensemble model
    ensemble = VotingClassifier([
        ('rf', models["randomforest"]),
        ('gb', models["gradient_boosting"]),
        ('svm', models["svm"])
    ], voting='soft')
    
    models["ensemble"] = ensemble
    
    results = {}
    best_model = None
    best_f1 = -1
    best_name = ""
    
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_features, y_train, cv=cv, scoring='f1_weighted')
        print(f"CV F1 scores: {cv_scores}")
        print(f"CV F1 mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train on full training set
        model.fit(X_train_features, y_train)
        
        # Predictions
        preds = model.predict(X_val_features)
        probs = model.predict_proba(X_val_features) if hasattr(model, 'predict_proba') else None
        
        # Metrics
        acc = accuracy_score(y_val, preds)
        f1w = f1_score(y_val, preds, average="weighted")
        precision, recall, f1, support = precision_recall_fscore_support(y_val, preds, average=None)
        
        print(f"{name} Results:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1-weighted: {f1w:.4f}")
        print(f"  Precision (fake/real): {precision}")
        print(f"  Recall (fake/real): {recall}")
        print(f"  F1 (fake/real): {f1}")
        
        results[name] = {
            "accuracy": acc,
            "f1_weighted": f1w,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist()
        }
        
        if f1w > best_f1:
            best_f1 = f1w
            best_model = model
            best_name = name
    
    # Save best model and artifacts
    print(f"\nSaving best model: {best_name} (F1={best_f1:.4f})")
    joblib.dump(feature_pipeline, "models/feature_pipeline.pkl")
    joblib.dump(best_model, "models/model_improved.pkl")
    joblib.dump(le, "models/label_encoder_improved.pkl")
    
    # Confusion matrix for best model
    preds_best = best_model.predict(X_val_features)
    cm = confusion_matrix(y_val, preds_best)
    plot_confusion_matrix(cm, list(le.classes_), "reports/confusion_matrix_improved.png")
    
    # Feature importance plot (if applicable)
    if hasattr(best_model, 'feature_importances_'):
        # Get feature names
        feature_names = []
        if hasattr(feature_pipeline.named_transformers_['tfidf'], 'get_feature_names_out'):
            tfidf_features = feature_pipeline.named_transformers_['tfidf'].get_feature_names_out()
            feature_names.extend(tfidf_features)
        
        linguistic_features = ['char_count', 'word_count', 'sentence_count', 'avg_word_length', 
                             'avg_sentence_length', 'exclamation_count', 'question_count', 
                             'caps_ratio', 'stopword_ratio', 'unique_word_ratio', 'positive_word_ratio', 
                             'negative_word_ratio', 'fake_indicator_ratio', 'number_count', 
                             'number_ratio', 'url_count', 'email_count']
        feature_names.extend(linguistic_features)
        
        plot_feature_importance(best_model, feature_names, "reports/feature_importance.png")
    
    # Save detailed metrics
    with open("reports/metrics_improved.json", "w") as f:
        json.dump({
            "results": results, 
            "best_model": best_name,
            "dataset_info": {
                "total_samples": int(len(df)),
                "fake_samples": int((y == "fake").sum()),
                "real_samples": int((y == "real").sum()),
                "class_balance": float((y == "fake").mean())
            }
        }, f, indent=2)
    
    print(f"\nBest model saved: {best_name}")
    print("Artifacts saved to models/ and reports/")
    print(f"Final F1-weighted score: {best_f1:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset.csv")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--max-features", type=int, default=20000)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()
    main(args)