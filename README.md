# Fake News Classifier - ML Project

An advanced machine learning system for detecting fake news articles with improved accuracy and comprehensive feature engineering.

## 🎯 Project Overview

This project implements a sophisticated fake news detection system using multiple machine learning algorithms and advanced text processing techniques. The system analyzes news articles and classifies them as either "fake" or "real" with high accuracy.

## 📊 Performance Metrics

### Current Best Model: Gradient Boosting Classifier
- **Overall Accuracy**: 72.3%
- **F1-Weighted Score**: 71.0%
- **Cross-Validation Score**: 69.3% (±3.8%)

### Class-Specific Performance
- **Fake News Detection**:
  - Precision: 75.0%
  - Recall: 86.1%
  - F1-Score: 80.2%

- **Real News Detection**:
  - Precision: 64.3%
  - Recall: 46.6%
  - F1-Score: 54.0%

## 🚀 Key Features

### Advanced Text Processing
- **Enhanced TF-IDF Vectorization**: Up to 20,000 features with trigrams
- **Linguistic Feature Extraction**: Word count, sentence length, punctuation analysis
- **Sentiment Analysis**: Positive/negative word detection
- **Fake News Indicators**: Detection of sensationalist language patterns
- **Text Statistics**: Character count, caps ratio, stopword analysis

### Multiple ML Algorithms
- Logistic Regression (with class balancing)
- Random Forest (300 estimators)
- Gradient Boosting (200 estimators)
- Support Vector Machine (RBF kernel)
- Naive Bayes (Multinomial)
- Ensemble Voting Classifier

### Robust Evaluation
- 5-fold Cross-Validation
- Stratified train/test splits
- Class balancing for imbalanced dataset
- Comprehensive metrics reporting

## 📁 Project Structure

```
ML-Fake-News-detector/
├── data/
│   ├── dataset.csv              # Processed dataset (830 articles)
│   └── raw/                     # Raw text files organized by category
├── models/
│   ├── model_improved.pkl       # Best trained model
│   ├── feature_pipeline.pkl     # Feature extraction pipeline
│   └── label_encoder_improved.pkl # Label encoder
├── reports/
│   ├── metrics_improved.json    # Detailed performance metrics
│   ├── confusion_matrix_improved.png
│   └── feature_importance.png
├── src/
│   ├── train_improved.py        # Enhanced training script
│   ├── train_baseline.py        # Original baseline training
│   ├── predict.py               # Prediction script
│   ├── prepare_data.py          # Data preprocessing
│   └── streamlit_app.py         # Web interface
├── requirements.txt
└── README.md
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
```bash
# Clone the repository
git clone <https://github.com/Ninad-18/ML-Fake-News-detector>
cd ML-Fake-News-detector

# Install dependencies
pip install -r requirements.txt
```

### Required Dependencies
```
pandas
numpy
scikit-learn
joblib
streamlit
nltk
matplotlib
seaborn
```

## 🚀 Usage

### Training the Model
```bash
# Train with improved features and algorithms
python src/train_improved.py

# Train baseline model (for comparison)
python src/train_baseline.py
```

### Making Predictions
```bash
# Predict using command line
python src/predict.py --text "Your news article text here"

# Launch enhanced web interface
streamlit run app.py

# Launch original web interface
streamlit run src/streamlit_app.py
```

### Data Preparation
```bash
# Process raw text files into CSV format
python src/prepare_data.py --raw-dir data/raw --out-csv data/dataset.csv
```

## 📈 Model Improvements

### Baseline vs Improved Performance
| Metric | Baseline | Improved | Improvement |
|--------|----------|----------|-------------|
| Accuracy | 66.4% | **72.3%** | **+5.9%** |
| F1-Score | 64.6% | **71.0%** | **+6.4%** |
| Features | 10K TF-IDF | **20K TF-IDF + Linguistic** | **+100%** |

### Key Enhancements
1. **Feature Engineering**: Added 17 linguistic features including sentiment analysis
2. **Algorithm Selection**: Tested 6 different ML algorithms with ensemble methods
3. **Text Processing**: Enhanced cleaning, trigrams, and normalization
4. **Evaluation**: Cross-validation and stratified sampling
5. **Class Balancing**: Handled dataset imbalance (65% fake, 35% real)

## 🎨 New Features Added 

### Enhanced Streamlit UI (`app.py`)
- **Professional Interface**: Modern, responsive design with custom CSS
- **Real-time Analysis**: Instant predictions with confidence scores
- **Model Explainability**: Shows top informative features for each prediction
- **Example Articles**: Built-in examples for testing
- **Performance Metrics**: Live display of model performance
- **Text Analysis**: Word count, character count, and linguistic features

### Model Explainability
- **Jupyter Notebook**: `model_explainability.ipynb` with detailed analysis
- **Top Features**: Identification of most informative words and patterns
- **Example Analysis**: Explanation of why articles are classified as fake/real
- **Feature Importance**: Understanding which linguistic features matter most

### Comprehensive Visualizations
- **ROC Curve**: `reports/roc_curve.png` - Model discrimination ability
- **Feature Importance**: `reports/feature_importance.png` - Top 25 most important features
- **Comprehensive Analysis**: `reports/comprehensive_analysis.png` - Multi-panel performance overview
- **Confusion Matrix**: `reports/confusion_matrix_improved.png` - Classification results

## 📊 Dataset Information

- **Total Articles**: 830
- **Fake News**: 540 articles (65%)
- **Real News**: 290 articles (35%)
- **Sources**: Multiple datasets including celebrity news, political news, and general articles
- **Format**: Cleaned text with metadata

## 🔧 Configuration

### Training Parameters
- **Validation Split**: 20%
- **Max Features**: 20,000
- **Cross-Validation**: 5-fold stratified
- **Random State**: 42 (for reproducibility)

### Model Hyperparameters
- **Gradient Boosting**: 200 estimators, learning rate 0.1, max depth 6
- **Random Forest**: 300 estimators, max depth 20
- **Logistic Regression**: C=0.1, balanced class weights

## 📝 Results Analysis

The improved model shows significant enhancements:
- **Better Fake News Detection**: 86.1% recall means it catches most fake articles
- **Improved Real News Detection**: 64.3% precision for real news classification
- **Robust Performance**: Consistent cross-validation scores across folds
- **Feature Importance**: TF-IDF features and linguistic patterns drive predictions

## 🚀 Future Enhancements

### Next Steps: Advanced AI Integration
- **BERT Embeddings**: Implement transformer-based embeddings for better semantic understanding
- **LLM Fine-tuning**: Fine-tune large language models (GPT, BERT) specifically for fake news detection
- **Deep Learning**: Implement LSTM/Transformer models with attention mechanisms
- **Word Embeddings**: Use Word2Vec, GloVe, or FastText embeddings
- **External Data**: Incorporate fact-checking databases and real-time verification APIs
- **Real-time Processing**: Deploy as API for live news classification
- **Multi-language Support**: Extend to other languages using multilingual models
- **Ensemble Methods**: Combine multiple models for even better accuracy
- **Active Learning**: Continuously improve with user feedback

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📞 Contact

For questions or suggestions, please open an issue in the repository.

---

**Note**: This model is for educational and research purposes. Always verify important news through multiple reliable sources.
