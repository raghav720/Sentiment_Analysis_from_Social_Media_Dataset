## 📝 Sentiment Analysis from Social Media Dataset

This project performs **multi-class sentiment classification** on social media text using TF-IDF vectorization, **SMOTE** for handling imbalanced classes, and various supervised machine learning models with **hyperparameter tuning** via `RandomizedSearchCV`.

---

### 📁 Dataset

| Column   | Description                                    |
|----------|------------------------------------------------|
| `Text`   | Raw user text                                  |
| `Sentiment` | Labeled emotion/sentiment (e.g., Happy, Angry, Love) |

**Input File:** `sentimentdataset (1).csv`

---

### 🔁 Sentiment Simplification

| Original Labels                    | Mapped To   |
|------------------------------------|-------------|
| Joy, Love, Excited, etc.           | **Positive** |
| Sad, Angry, Fear, etc.             | **Negative** |
| Surprise, Confusion, etc.          | **Neutral**  |

---

### 🧪 Workflow

✅ **1. Preprocessing**  
• Stripped whitespace from sentiment labels  
• Mapped detailed emotions into 3 main classes (Positive, Negative, Neutral)

✅ **2. Feature Engineering**  
• Applied `TfidfVectorizer` with n-grams (trigram), `max_features=6000`, `min_df=3`, `max_df=0.6`  
• Converted text into numerical feature vectors

✅ **3. Label Encoding**  
• Encoded target labels into integers using `LabelEncoder`

✅ **4. Handle Imbalance**  
• Used `SMOTE` to oversample minority classes and balance the dataset

✅ **5. Train-Test Split**  
• Splitted data into 80% train / 20% test with **stratification**

✅ **6. Hyperparameter Tuning**  
Applied `RandomizedSearchCV` for the following classifiers:

| Model               | Parameters Tuned (Examples)      |
|---------------------|----------------------------------|
| Random Forest        | `n_estimators`, `max_depth`      |
| Logistic Regression  | `C`, `penalty`                   |
| SVM                  | `C`, `gamma`, `kernel`           |
| Decision Tree        | `max_depth`, `min_samples_split` |

> _(Optional)_ Other models like **Naive Bayes**, **XGBoost**, **KNN** can also be added

---

### 📊 Evaluation Metrics

- Classification report  
- **F1-score**  
- **Recall / Precision**  
- Confusion Matrix  
- ROC Curve & AUC (where applicable)

---

### 🎯 Goal

To **identify the best-performing model** for sentiment classification across three categories (**Positive**, **Negative**, **Neutral**), ensuring robust performance even on **imbalanced data**.

---

### 📚 Libraries Used

pandas, numpy, matplotlib, seaborn
sklearn: preprocessing, models, metrics, cross-validation
xgboost, imblearn, scipy.stats
