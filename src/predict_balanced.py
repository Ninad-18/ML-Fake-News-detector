import joblib
import argparse
import numpy as np
import pandas as pd
import sys
import os

# Add the src directory to path to import the LinguisticFeatures class
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the LinguisticFeatures class from the training script
from src.train_balanced import LinguisticFeatures

def load_artifacts():
    """Load the balanced model and preprocessing pipeline"""
    try:
        model = joblib.load("models/model_balanced.pkl")
        feature_pipeline = joblib.load("models/feature_pipeline_balanced.pkl")
        label_encoder = joblib.load("models/label_encoder_balanced.pkl")
        optimal_threshold = joblib.load("models/optimal_threshold_balanced.pkl")
        return model, feature_pipeline, label_encoder, optimal_threshold
    except FileNotFoundError:
        print("Balanced model files not found. Please run train_balanced.py first.")
        return None, None, None, None

def predict_with_threshold(text, model, feature_pipeline, label_encoder, threshold):
    """Make prediction using optimal threshold"""
    # Transform text
    text_df = pd.DataFrame({'text': [text]})
    features = feature_pipeline.transform(text_df)
    
    if hasattr(model, 'predict_proba'):
        # Get probabilities
        probabilities = model.predict_proba(features)[0]
        real_prob = probabilities[1]  # Probability of real news
        
        # Use optimal threshold
        prediction = 1 if real_prob >= threshold else 0
        confidence = max(probabilities)
        
        return prediction, probabilities, confidence, real_prob
    else:
        # Fallback for models without predict_proba
        prediction = model.predict(features)[0]
        probabilities = None
        confidence = None
        real_prob = None
        return prediction, probabilities, confidence, real_prob

def test_multiple_thresholds(text, model, feature_pipeline, label_encoder):
    """Test multiple thresholds to find the best balance"""
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    print(f"Testing multiple thresholds for: {text[:50]}...")
    print("=" * 80)
    
    for threshold in thresholds:
        prediction, probabilities, confidence, real_prob = predict_with_threshold(
            text, model, feature_pipeline, label_encoder, threshold
        )
        
        label = label_encoder.inverse_transform([prediction])[0]
        
        print(f"Threshold {threshold:.1f}: {label.upper()} (Real prob: {real_prob:.3f})")
    
    print("=" * 80)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", nargs="+", help="One or more texts", required=True)
    ap.add_argument("--test-thresholds", action="store_true", help="Test multiple thresholds")
    args = ap.parse_args()
    
    # Load model artifacts
    model, feature_pipeline, label_encoder, optimal_threshold = load_artifacts()
    
    if model is None:
        exit(1)
    
    texts = [" ".join(args.text)]
    
    if args.test_thresholds:
        test_multiple_thresholds(texts[0], model, feature_pipeline, label_encoder)
    else:
        print(f"Using balanced threshold: {optimal_threshold:.3f}")
        print("=" * 60)
        
        for i, text in enumerate(texts):
            prediction, probabilities, confidence, real_prob = predict_with_threshold(
                text, model, feature_pipeline, label_encoder, optimal_threshold
            )
            
            label = label_encoder.inverse_transform([prediction])[0]
            
            print(f"Text: {text[:100]}...")
            print(f"Prediction: {label.upper()}")
            
            if probabilities is not None:
                print(f"Probabilities: Fake={probabilities[0]:.3f}, Real={probabilities[1]:.3f}")
                print(f"Real News Probability: {real_prob:.3f}")
                print(f"Threshold Used: {optimal_threshold:.3f}")
                print(f"Confidence: {confidence:.3f}")
            
            print("-" * 60)