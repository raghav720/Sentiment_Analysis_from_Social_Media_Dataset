# Sentiment Analysis from Social Media Dataset

This project performs multi-class sentiment classification on text data using TF-IDF vectorization, SMOTE for handling imbalanced classes, and various supervised machine learning models with hyperparameter tuning via RandomizedSearchCV.

# 📁 Dataset

Input CSV: sentimentdataset (1).csv

Columns:
  Text: Raw user text
  Sentiment: Labeled emotion/sentiment (e.g., Happy, Angry, Love)

#Sentiment Simplification

Original labels are mapped to:

Positive: Joy, Love, Excited, etc.
Negative: Sad, Angry, Fear, etc.
Neutral: Surprise, Confusion, etc.

#🧪 Workflow

✅ 1. Preprocessing
Strip whitespace from sentiment labels.
Map detailed sentiments into 3 main classes: Positive, Negative, Neutral.
✅ 2. Feature Engineering
Use TfidfVectorizer with trigrams, max 6000 features, min_df=3, max_df=0.6.
Convert text into numerical feature vectors.
✅ 3. Label Encoding
Convert target labels (3-class) into numerical values using LabelEncoder.
✅ 4. Handle Imbalance
Apply SMOTE to oversample the minority classes and balance the dataset.
✅ 5. Train-Test Split
Split into 80% training and 20% testing data with stratification.
✅ 6. Hyperparameter Tuning
Use RandomizedSearchCV for the following classifiers:

Random Forest
Logistic Regression
SVM
Decision Tree
(Others like Naive Bayes, XGBoost, KNN can be optionally added)
Each model uses a parameter grid (e.g., n_estimators, max_depth, C, gamma, etc.).

#📊 Evaluation Metrics

Each tuned model is evaluated using:

Classification Report
F1-Score
Recall
Confusion Matrix
ROC Curve & AUC (if applicable)

#🚀 Goal

To identify the best model for text sentiment classification across three categories, ensuring performance is robust even with imbalanced classes.

#📚 Libraries Used
pandas, numpy, matplotlib, seaborn
sklearn: preprocessing, models, metrics, cross-validation
xgboost, imblearn, scipy.stats
