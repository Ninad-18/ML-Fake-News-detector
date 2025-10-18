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
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, precision_recall_curve, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

import matplotlib.pyplot as plt
import seaborn as sns

class LinguisticFeatures(BaseEstimator, TransformerMixin):
    """Extract linguistic features optimized for balanced fake/real news detection"""
    
    def __init__(self):
        # Simple stop words list
        self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
    
    def extract_features(self, text):
        """Extract balanced linguistic features"""
        features = {}
        
        # Basic text statistics
        words = text.lower().split()
        sentences = text.split('.')
        
        features['char_count'] = len(text)
        features['word_count'] = len(words)
        features['sentence_count'] = len(sentences)
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        features['avg_sentence_length'] = np.mean([len(sent.split()) for sent in sentences]) if sentences else 0
        
        # Punctuation and special characters (key indicators)
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        # Word-level features
        features['stopword_ratio'] = sum(1 for word in words if word in self.stop_words) / len(words) if words else 0
        features['unique_word_ratio'] = len(set(words)) / len(words) if words else 0
        
        # Sentiment indicators
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'positive', 'success', 'win', 'victory', 'best', 'better', 'improve', 'increase', 'rise', 'gain', 'development', 'progress', 'achievement', 'growth', 'advancement', 'innovation', 'breakthrough'}
        negative_words = {'bad', 'terrible', 'awful', 'horrible', 'negative', 'fail', 'loss', 'defeat', 'crisis', 'problem', 'worst', 'worse', 'decline', 'decrease', 'fall', 'drop', 'disaster', 'scandal', 'corruption', 'fraud', 'deception'}
        
        features['positive_word_ratio'] = sum(1 for word in words if word in positive_words) / len(words) if words else 0
        features['negative_word_ratio'] = sum(1 for word in words if word in negative_words) / len(words) if words else 0
        
        # Strong fake news indicators
        fake_indicators = {'breaking', 'shocking', 'unbelievable', 'exclusive', 'urgent', 'alert', 'warning', 'conspiracy', 'cover-up', 'scandal', 'exposed', 'revealed', 'leaked', 'bombshell', 'explosive', 'outrageous', 'incredible', 'amazing', 'unprecedented', 'stunning', 'devastating', 'catastrophic'}
        features['fake_indicator_ratio'] = sum(1 for word in words if word in fake_indicators) / len(words) if words else 0
        
        # Emotional language indicators (fake news often uses emotional triggers)
        emotional_words = {'outrageous', 'devastating', 'shocking', 'stunning', 'incredible', 'unbelievable', 'terrifying', 'horrifying', 'appalling', 'disgusting', 'revolting', 'disturbing', 'alarming', 'frightening', 'scary'}
        features['emotional_word_ratio'] = sum(1 for word in words if word in emotional_words) / len(words) if words else 0
        
        # Numbers and statistics (real news often has more data)
        features['number_count'] = len(re.findall(r'\d+', text))
        features['number_ratio'] = features['number_count'] / len(words) if words else 0
        
        # URL and email indicators
        features['url_count'] = len(re.findall(r'http\S+|www\.\S+', text))
        features['email_count'] = len(re.findall(r'\S+@\S+', text))
        
        # Professional language indicators (strong real news signals)
        professional_words = {'according', 'reported', 'official', 'statement', 'announced', 'confirmed', 'data', 'research', 'study', 'analysis', 'government', 'department', 'ministry', 'institution', 'organization', 'agency', 'commission', 'committee', 'council', 'authority', 'spokesperson', 'representative'}
        features['professional_word_ratio'] = sum(1 for word in words if word in professional_words) / len(words) if words else 0
        
        # Formal language indicators
        formal_words = {'however', 'therefore', 'furthermore', 'moreover', 'consequently', 'nevertheless', 'additionally', 'specifically', 'particularly', 'especially', 'significantly', 'substantially', 'considerably', 'notably', 'remarkably'}
        features['formal_word_ratio'] = sum(1 for word in words if word in formal_words) / len(words) if words else 0
        
        # Source attribution indicators
        source_words = {'source', 'sources', 'official', 'spokesperson', 'representative', 'spokesman', 'spokeswoman', 'spokespeople', 'authority', 'expert', 'specialist', 'analyst', 'researcher', 'scientist', 'professor', 'doctor', 'director'}
        features['source_word_ratio'] = sum(1 for word in words if word in source_words) / len(words) if words else 0
        
        # Temporal indicators (real news often has specific time references)
        temporal_words = {'today', 'yesterday', 'tomorrow', 'recently', 'previously', 'earlier', 'later', 'currently', 'now', 'then', 'when', 'while', 'during', 'after', 'before', 'since', 'until'}
        features['temporal_word_ratio'] = sum(1 for word in words if word in temporal_words) / len(words) if words else 0
        
        # Specificity indicators (real news uses specific details)
        specific_words = {'percent', 'percentage', 'million', 'billion', 'thousand', 'hundred', 'dozen', 'several', 'many', 'few', 'most', 'some', 'all', 'each', 'every', 'specific', 'particular', 'certain'}
        features['specificity_word_ratio'] = sum(1 for word in words if word in specific_words) / len(words) if words else 0
        
        # Question indicators (fake news often uses rhetorical questions)
        features['question_indicators'] = sum(1 for word in words if word in {'what', 'why', 'how', 'when', 'where', 'who', 'which'}) / len(words) if words else 0
        
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

def find_balanced_threshold(y_true, y_proba):
    """Find threshold that balances both classes well"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Calculate F1 scores for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # Find threshold that maximizes F1 score
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    return optimal_threshold, f1_scores[optimal_idx]

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
    
    # Balanced models with moderate class weights
    models = {
        "logistic": LogisticRegression(
            max_iter=2000, 
            class_weight={0: 0.85, 1: 1.15},  # Slight favor for real news
            solver="saga", 
            n_jobs=-1, 
            random_state=args.random_state,
            C=0.1
        ),
        "randomforest": RandomForestClassifier(
            n_estimators=300, 
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1, 
            random_state=args.random_state,
            class_weight={0: 0.85, 1: 1.15}  # Slight favor for real news
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
            class_weight={0: 0.85, 1: 1.15},  # Slight favor for real news
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
    best_threshold = 0.5
    
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
        
        # Get probabilities for threshold tuning
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_val_features)[:, 1]  # Probability of real news
            
            # Find balanced threshold
            optimal_threshold, optimal_f1 = find_balanced_threshold(y_val, y_proba)
            print(f"Optimal threshold: {optimal_threshold:.3f} (F1: {optimal_f1:.3f})")
            
            # Predictions with optimal threshold
            y_pred_thresh = (y_proba >= optimal_threshold).astype(int)
            
            # Also get standard predictions
            y_pred = model.predict(X_val_features)
        else:
            y_pred = model.predict(X_val_features)
            y_pred_thresh = y_pred
            optimal_threshold = 0.5
        
        # Metrics with optimal threshold
        acc_thresh = accuracy_score(y_val, y_pred_thresh)
        f1w_thresh = f1_score(y_val, y_pred_thresh, average="weighted")
        precision_thresh, recall_thresh, f1_thresh, support_thresh = precision_recall_fscore_support(y_val, y_pred_thresh, average=None)
        
        # Standard metrics
        acc = accuracy_score(y_val, y_pred)
        f1w = f1_score(y_val, y_pred, average="weighted")
        precision, recall, f1, support = precision_recall_fscore_support(y_val, y_pred, average=None)
        
        print(f"{name} Results (with optimal threshold):")
        print(f"  Accuracy: {acc_thresh:.4f}")
        print(f"  F1-weighted: {f1w_thresh:.4f}")
        print(f"  Precision (fake/real): {precision_thresh}")
        print(f"  Recall (fake/real): {recall_thresh}")
        print(f"  F1 (fake/real): {f1_thresh}")
        print(f"  Optimal Threshold: {optimal_threshold:.3f}")
        
        results[name] = {
            "accuracy": acc_thresh,
            "f1_weighted": f1w_thresh,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "precision": precision_thresh.tolist(),
            "recall": recall_thresh.tolist(),
            "f1": f1_thresh.tolist(),
            "optimal_threshold": optimal_threshold,
            "standard_accuracy": acc,
            "standard_f1": f1w
        }
        
        if f1w_thresh > best_f1:
            best_f1 = f1w_thresh
            best_model = model
            best_name = name
            best_threshold = optimal_threshold
    
    # Save best model and artifacts
    print(f"\nSaving best model: {best_name} (F1={best_f1:.4f}, Threshold={best_threshold:.3f})")
    joblib.dump(feature_pipeline, "models/feature_pipeline_balanced.pkl")
    joblib.dump(best_model, "models/model_balanced.pkl")
    joblib.dump(le, "models/label_encoder_balanced.pkl")
    joblib.dump(best_threshold, "models/optimal_threshold_balanced.pkl")
    
    # Confusion matrix for best model with optimal threshold
    if hasattr(best_model, 'predict_proba'):
        y_proba_best = best_model.predict_proba(X_val_features)[:, 1]
        preds_best = (y_proba_best >= best_threshold).astype(int)
    else:
        preds_best = best_model.predict(X_val_features)
    
    cm = confusion_matrix(y_val, preds_best)
    plot_confusion_matrix(cm, list(le.classes_), "reports/confusion_matrix_balanced.png")
    
    # Save detailed metrics
    with open("reports/metrics_balanced.json", "w") as f:
        json.dump({
            "results": results, 
            "best_model": best_name,
            "optimal_threshold": best_threshold,
            "dataset_info": {
                "total_samples": int(len(df)),
                "fake_samples": int((y == "fake").sum()),
                "real_samples": int((y == "real").sum()),
                "class_balance": float((y == "fake").mean())
            }
        }, f, indent=2)
    
    print(f"\nBest model saved: {best_name}")
    print(f"Optimal threshold: {best_threshold:.3f}")
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

