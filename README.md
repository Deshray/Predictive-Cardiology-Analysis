# 🩺 Heart Disease Prediction using Machine Learning

This project builds and evaluates multiple machine learning models to predict the presence of heart disease based on patient symptoms and clinical measurements. It includes data exploration, feature visualizations, and classification model comparison to determine the most accurate predictor.

---

## 📊 Dataset

- Source: `heart.csv` dataset
- 1190 patient records
- Features include Age, Cholesterol, Chest Pain Type, ST Slope, Blood Pressure, etc.

---

## 🧠 Machine Learning Models

The following models were trained and evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

Models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score

---

## 📈 Visualizations

This project includes:
- Pie charts, bar plots, and histograms
- Box plots and strip plots for feature relationships
- ROC curves for all models
- Zoomed-in performance comparison bar chart
- Feature importance plot (Random Forest)

---

## ✅ Key Results

Top 3 models based on performance:
1. **Support Vector Machine (AUC: 0.949)** – Highest recall and ROC-AUC
2. **Gradient Boosting (AUC: 0.936)** – Best precision, strong F1
3. **Random Forest (AUC: 0.942)** – Most balanced performance

> Cross-Validation Accuracy (Random Forest): ~88%

---

## 💡 Real-World Relevance

This model can assist clinicians in early heart disease screening by identifying high-risk patients based on non-invasive clinical data. It demonstrates the application of machine learning in healthcare diagnostics.

---

## ⚖️ Ethical Considerations

While promising, this model must be evaluated on larger and more diverse datasets to ensure fairness across age, gender, and demographic groups. These models should support — not replace — clinical judgment.

---

## 🧾 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/heart-disease-ml.git
   cd heart-disease-ml
