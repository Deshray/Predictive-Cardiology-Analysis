import numpy as np
import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import time

warnings.filterwarnings("ignore")
pd.set_option("display.max_rows", None)

# Load dataset
file_path = r'C:\Users\Dibyendu Kumar Ray\Downloads\heart.csv'
hd = pd.read_csv(file_path)
print(hd.head())

# Use red and yellow for consistency
colors = ['#FDD20E', '#F93822']

# Correlation matrix
numeric_columns = hd.select_dtypes(include=np.number)
fig = px.imshow(numeric_columns.corr(),
                title="Correlation Plot of the Heart Failure Prediction",
                color_continuous_scale=['#FDD20E', 'white', '#F93822'])
fig.show()

# Identify categorical and numerical features
col = list(hd.columns)
categorical_features = [i for i in col if len(hd[i].unique()) <= 6]
numerical_features = [i for i in col if i not in categorical_features]

print('Categorical Features:', *categorical_features)
print('Numerical Features:', *numerical_features)

# Pie chart for Heart Disease distribution
plt.figure(figsize=(10, 5))
circle = [hd['HeartDisease'].value_counts()[1], hd['HeartDisease'].value_counts()[0]]
plt.pie(circle, labels=['Heart Disease', 'No Heart Disease'], autopct='%1.1f%%', startangle=90, explode=(0.1, 0),
        colors=colors, wedgeprops={'edgecolor': 'black', 'linewidth': 1})
plt.title('Heart Disease Distribution')
plt.show()

# --- 1. Categorical Feature Distributions ---
for feature in categorical_features:
    plt.figure(figsize=(8, 4))
    unique_vals = len(hd[feature].unique())
    palette = sns.color_palette("Set2", unique_vals) if unique_vals > 2 else colors[::-1]
    sns.countplot(x=feature, data=hd, palette=palette, edgecolor='black')
    plt.title(f'{feature} Distribution')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.show()

# --- 2. Categorical vs HeartDisease Counts ---
for feature in categorical_features:
    plt.figure(figsize=(8, 4))
    ax = sns.countplot(x=feature, data=hd, hue="HeartDisease", palette=colors[::-1], edgecolor='black')
    for rect in ax.patches:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height + 3, f"{(height / len(hd) * 100):.1f}%", ha='center')
    plt.title(f'{feature} vs HeartDisease')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.legend(['No Heart Disease', 'Heart Disease'])
    plt.show()

# --- 3. Pie Charts for HeartDisease Positive Group ---
def pie_chart(data, feature, labels):
    plt.figure(figsize=(6, 6))
    values = data[feature].value_counts(normalize=True) * 100
    unique_vals = len(values)
    pie_colors = sns.color_palette("Set2", unique_vals) if unique_vals > 2 else colors[::-1]
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90,
            explode=[0.1 if i == 0 else 0 for i in range(len(values))], colors=pie_colors,
            wedgeprops={'edgecolor': 'black', 'linewidth': 1})
    plt.title(f'{feature} Distribution (Heart Disease)')
    plt.show()

hd_pos = hd[hd['HeartDisease'] == 1]
pie_chart(hd_pos, 'Sex', ['Male', 'Female'])
pie_chart(hd_pos, 'FastingBS', ['FBS < 120 mg/dl', 'FBS > 120 mg/dl'])
pie_chart(hd_pos, 'ExerciseAngina', ['Angina', 'No Angina'])
pie_chart(hd_pos, 'ChestPainType', ['ASY', 'NAP', 'ATA', 'TA'])
pie_chart(hd_pos, 'RestingECG', ['Normal', 'ST', 'LVH'])
pie_chart(hd_pos, 'ST_Slope', ['Flat', 'Up', 'Down'])

# --- 4. Numerical Features vs Heart Disease ---
for feature in numerical_features:
    plt.figure(figsize=(12, 6))
    ax = sns.histplot(data=hd, x=feature, hue='HeartDisease', multiple='stack', palette=colors[::-1], edgecolor='black', binwidth=1)
    plt.title(f'{feature} vs Heart Disease')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.legend(['No Heart Disease', 'Heart Disease'])
    plt.show()

# --- 5. Boxplots of numerical features by HeartDisease ---
sns.set(style="whitegrid")
for i in numerical_features:
    plt.figure(figsize=(10, 6)) 
    sns.boxplot(x='HeartDisease', y=i, data=hd, palette=colors[::-1])
    plt.title(f'Box Plot of {i} by HeartDisease')
    plt.xlabel('Heart Disease')
    plt.ylabel(i)
    plt.show()

# --- 6. Strip plots of numerical features vs categorical features ---
cat_feature_names = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
for cat in cat_feature_names:
    for num in numerical_features:
        plt.figure(figsize=(10, 6))
        sns.stripplot(x=cat, y=num, data=hd, hue='HeartDisease', palette=colors[::-1], dodge=False, jitter=True, alpha=0.6)
        plt.title(f'{num} vs {cat}')
        plt.legend(['No Heart Disease', 'Heart Disease'])
        plt.show()

# Encode categorical variables
for col in categorical_features:
    le = LabelEncoder()
    hd[col] = le.fit_transform(hd[col])

# Prepare ML modeling
X = hd.drop('HeartDisease', axis=1)
y = hd['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(probability=True, random_state=42)
}

results_df = pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score", "AUC"])
roc_curves = []  # To collect ROC data for final combined plot

for name, model in models.items():
    print(f"\n{'='*30}\n{name} Results:\n{'='*30}")
    time.sleep(1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    if y_proba is not None:
        print(f"\nROC-AUC Score: {auc:.2f}")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})", color=colors[1])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {name}")
        plt.legend()
        plt.show()
        roc_curves.append((name, fpr, tpr, auc))

    results_df.loc[len(results_df)] = {
        "Model": name,
        "Accuracy": round(accuracy, 3),
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "F1-Score": round(f1, 3),
        "AUC": round(auc, 3) if auc else "N/A"
    }

    time.sleep(2)

# Feature Importance Visualization (Random Forest)
feature_names = hd.drop('HeartDisease', axis=1).columns
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

importances = rf_model.feature_importances_
sorted_indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[sorted_indices][:10],
            y=feature_names[sorted_indices][:10],
            palette='Reds_r')
plt.title("Top 10 Most Important Features (Random Forest)")
plt.xlabel("Feature Importance Score")
plt.ylabel("Feature")
plt.grid(True)
plt.tight_layout()
plt.show()

# Final Comparison Table
print("\nModel Comparison Results:")
print(results_df.to_string(index=False))
cv_score = cross_val_score(RandomForestClassifier(random_state=42), X, y, cv=5)
print(f"\nCross-Validation Accuracy (Random Forest): {cv_score.mean():.3f}")

# Bar Chart Comparison
results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-Score"]].plot(kind="bar", figsize=(10, 6), color=['#F93822', '#FDD20E', '#0077B6', '#2ECC71'])
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-Score"]].plot(kind="bar", figsize=(10, 6), color=['#F93822', '#FDD20E', '#0077B6', '#2ECC71'])
plt.title("Model Performance Comparison (Zoomed In)")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.ylim(0.7, 0.95)
plt.yticks(np.arange(0.7, 0.951, 0.02))
plt.show()

# Combined ROC Curve Plot
plt.figure(figsize=(10, 7))
for name, fpr, tpr, auc in roc_curves:
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - All Models")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()