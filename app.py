import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

# Page configuration
st.set_page_config(
    page_title="Fake News Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .fake-prediction {
        border-left-color: #ff6b6b;
        background-color: #ffe0e0;
    }
    .real-prediction {
        border-left-color: #51cf66;
        background-color: #e0ffe0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .stTextArea > div > div > textarea {
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

class LinguisticFeatures(BaseEstimator, TransformerMixin):
    """Extract linguistic features from text"""
    
    def __init__(self):
        self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
    
    def extract_features(self, text):
        """Extract various linguistic features"""
        features = {}
        
        words = text.lower().split()
        sentences = text.split('.')
        
        features['char_count'] = len(text)
        features['word_count'] = len(words)
        features['sentence_count'] = len(sentences)
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        features['avg_sentence_length'] = np.mean([len(sent.split()) for sent in sentences]) if sentences else 0
        
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        features['stopword_ratio'] = sum(1 for word in words if word in self.stop_words) / len(words) if words else 0
        features['unique_word_ratio'] = len(set(words)) / len(words) if words else 0
        
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'positive', 'success', 'win', 'victory', 'best', 'better', 'improve', 'increase', 'rise', 'gain'}
        negative_words = {'bad', 'terrible', 'awful', 'horrible', 'negative', 'fail', 'loss', 'defeat', 'crisis', 'problem', 'worst', 'worse', 'decline', 'decrease', 'fall', 'drop'}
        
        features['positive_word_ratio'] = sum(1 for word in words if word in positive_words) / len(words) if words else 0
        features['negative_word_ratio'] = sum(1 for word in words if word in negative_words) / len(words) if words else 0
        
        fake_indicators = {'breaking', 'shocking', 'unbelievable', 'exclusive', 'urgent', 'alert', 'warning', 'conspiracy', 'cover-up', 'scandal', 'exposed', 'revealed', 'leaked'}
        features['fake_indicator_ratio'] = sum(1 for word in words if word in fake_indicators) / len(words) if words else 0
        
        features['number_count'] = len(re.findall(r'\d+', text))
        features['number_ratio'] = features['number_count'] / len(words) if words else 0
        
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

@st.cache_resource
def load_model():
    """Load the trained model and preprocessing pipeline"""
    try:
        model = joblib.load("models/model_improved.pkl")
        feature_pipeline = joblib.load("models/feature_pipeline.pkl")
        label_encoder = joblib.load("models/label_encoder_improved.pkl")
        return model, feature_pipeline, label_encoder
    except FileNotFoundError:
        st.error("Model files not found. Please run the training script first.")
        return None, None, None

def get_top_features(text, model, feature_pipeline, top_n=10):
    """Extract top informative features for the given text"""
    try:
        # Transform the text
        text_df = pd.DataFrame({'text': [text]})
        features = feature_pipeline.transform(text_df)
        
        # Convert sparse matrix to dense if needed
        if hasattr(features, 'toarray'):
            features_dense = features.toarray()[0]
        else:
            features_dense = features[0]
        
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
        
        # Get feature importance if available
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = list(zip(feature_names, importances, features_dense))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            return feature_importance[:top_n]
        
        return None
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">📰 Fake News Classifier</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced ML-powered detection system with 72.3% accuracy")
    
    # Load model
    model, feature_pipeline, label_encoder = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar with model info
    with st.sidebar:
        st.header("📊 Model Information")
        st.metric("Accuracy", "72.3%")
        st.metric("F1-Score", "71.0%")
        st.metric("Fake News Recall", "86.1%")
        st.metric("Real News Precision", "64.3%")
        
        st.header("🔍 Model Details")
        st.write("**Algorithm**: Gradient Boosting")
        st.write("**Features**: 20K TF-IDF + 17 Linguistic")
        st.write("**Training Samples**: 830 articles")
        st.write("**Cross-Validation**: 5-fold")
        
        st.header("📈 Performance")
        st.write("**Fake News Detection**:")
        st.write("- Precision: 75.0%")
        st.write("- Recall: 86.1%")
        st.write("- F1-Score: 80.2%")
        
        st.write("**Real News Detection**:")
        st.write("- Precision: 64.3%")
        st.write("- Recall: 46.6%")
        st.write("- F1-Score: 54.0%")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔍 Analyze News Article")
        
        # Text input
        text_input = st.text_area(
            "Enter or paste your news headline/article:",
            height=200,
            placeholder="Paste your news article or headline here to analyze..."
        )
        
        # Prediction button
        if st.button("🚀 Analyze Article", type="primary", use_container_width=True):
            if not text_input.strip():
                st.warning("Please enter some text to analyze.")
            else:
                try:
                    # Make prediction
                    text_df = pd.DataFrame({'text': [text_input]})
                    features = feature_pipeline.transform(text_df)
                    prediction = model.predict(features)[0]
                    probability = model.predict_proba(features)[0]
                    
                    # Get prediction label
                    label = label_encoder.inverse_transform([prediction])[0]
                    confidence = float(probability.max())
                    
                    # Display prediction
                    prediction_class = "fake-prediction" if label == "fake" else "real-prediction"
                    prediction_emoji = "🚨" if label == "fake" else "✅"
                    
                    st.markdown(f"""
                    <div class="prediction-box {prediction_class}">
                        <h2>{prediction_emoji} Prediction: {label.upper()}</h2>
                        <h3>Confidence: {confidence:.1%}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show probability breakdown
                    col_prob1, col_prob2 = st.columns(2)
                    with col_prob1:
                        st.metric("Fake News Probability", f"{probability[0]:.1%}")
                    with col_prob2:
                        st.metric("Real News Probability", f"{probability[1]:.1%}")
                    
                    # Feature analysis
                    st.header("🔬 Model Explainability")
                    top_features = get_top_features(text_input, model, feature_pipeline)
                    
                    if top_features:
                        st.subheader("Top Informative Features:")
                        for i, (feature_name, importance, value) in enumerate(top_features[:5]):
                            st.write(f"{i+1}. **{feature_name}**: {value:.4f} (importance: {importance:.4f})")
                    else:
                        st.subheader("Feature Analysis:")
                        st.write("🔍 **Text Characteristics:**")
                        
                        # Simple text analysis
                        words = text_input.split()
                        sentences = text_input.split('.')
                        exclamation_count = text_input.count('!')
                        caps_count = sum(1 for c in text_input if c.isupper())
                        
                        fake_indicators = ['breaking', 'shocking', 'urgent', 'exclusive', 'alert', 'warning', 'conspiracy']
                        fake_words_found = [word.lower() for word in words if word.lower() in fake_indicators]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"• **Word Count**: {len(words)}")
                            st.write(f"• **Exclamation Marks**: {exclamation_count}")
                            st.write(f"• **Capital Letters**: {caps_count}")
                        with col2:
                            st.write(f"• **Sentence Count**: {len(sentences)}")
                            st.write(f"• **Fake Indicators**: {len(fake_words_found)}")
                            if fake_words_found:
                                st.write(f"• **Found Words**: {', '.join(fake_words_found)}")
                        
                        # Add explanation
                        st.write("")
                        st.write("💡 **Why this prediction?**")
                        if label == "fake":
                            if fake_words_found:
                                st.write(f"• Contains fake news indicators: {', '.join(fake_words_found)}")
                            if exclamation_count > 2:
                                st.write(f"• High exclamation count ({exclamation_count}) suggests sensationalism")
                            if caps_count > len(text_input) * 0.1:
                                st.write("• Excessive capitalization indicates emotional language")
                        else:
                            st.write("• Professional, factual language")
                            st.write("• No sensationalist indicators found")
                            st.write("• Balanced tone and structure")
                    
                    # Text analysis
                    st.subheader("📊 Text Analysis")
                    words = text_input.split()
                    sentences = text_input.split('.')
                    
                    col_ana1, col_ana2, col_ana3 = st.columns(3)
                    with col_ana1:
                        st.metric("Word Count", len(words))
                    with col_ana2:
                        st.metric("Character Count", len(text_input))
                    with col_ana3:
                        st.metric("Sentence Count", len(sentences))
                    
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
    
    with col2:
        st.header("📚 Example Articles")
        
        st.subheader("Real News Examples")
        real_examples = [
            "UK retail sales shrug off Brexit fears with February rise",
            "Scientists discover new method for renewable energy storage",
            "Local community raises funds for children's hospital"
        ]
        
        for example in real_examples:
            if st.button(f"📄 {example[:50]}...", key=f"real_{example[:10]}"):
                st.session_state.example_text = example
        
        st.subheader("Fake News Examples")
        fake_examples = [
            "BREAKING: Obama secretly controls Washington Post",
            "SHOCKING: Aspartame causes epidemic of MS and Lupus",
            "URGENT: Government hiding alien contact evidence"
        ]
        
        for example in fake_examples:
            if st.button(f"📄 {example[:50]}...", key=f"fake_{example[:10]}"):
                st.session_state.example_text = example
        
        # Load example if selected
        if hasattr(st.session_state, 'example_text'):
            st.text_area("Selected Example:", value=st.session_state.example_text, height=100)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🤖 Powered by Advanced Machine Learning | Accuracy: 72.3% | F1-Score: 71.0%</p>
        <p><em>Note: This tool is for educational purposes. Always verify important news through multiple reliable sources.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
