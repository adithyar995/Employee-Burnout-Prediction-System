# ==========================================
# Employee Burnout Prediction System
# Training Model
# ==========================================

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# Create Folders
# ==========================================

os.makedirs("model", exist_ok=True)
os.makedirs("graphs", exist_ok=True)

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("dataset/employee_burnout.csv")

print("\n========== DATASET LOADED ==========")
print(df.head())

print("\nDataset Shape :", df.shape)

print("\n========== DATASET INFO ==========")
print(df.info())

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE VALUES ==========")
print(df.duplicated().sum())

# ==========================================
# Label Encoding
# ==========================================

label_encoders = {}

categorical_columns = [
    "gender",
    "job_role",
    "burnout_level"
]

for column in categorical_columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    label_encoders[column] = encoder

print("\nLabel Encoding Completed Successfully!")

# ==========================================
# Features and Target
# ==========================================

X = df.drop("burnout_level", axis=1)
y = df["burnout_level"]

print("\nFeatures Shape :", X.shape)
print("Target Shape :", y.shape)

# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTraining Data :", X_train.shape)
print("Testing Data :", X_test.shape)
# ==========================================
# Build Random Forest Model
# ==========================================

print("\n========== TRAINING MODEL ==========")

model = RandomForestClassifier(
    n_estimators=40,
    max_depth=6,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    bootstrap=True,
    random_state=42
)

model.fit(X_train, y_train)

print("Model Training Completed Successfully!")

# ==========================================
# Prediction
# ==========================================

print("\n========== MAKING PREDICTIONS ==========")

y_pred = model.predict(X_test)

# ==========================================
# Model Performance
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

print("\n========== MODEL PERFORMANCE ==========")

print(f"Accuracy  : {accuracy*100:.2f}%")
print(f"Precision : {precision*100:.2f}%")
print(f"Recall    : {recall*100:.2f}%")
print(f"F1 Score  : {f1*100:.2f}%")

# ==========================================
# Confusion Matrix
# ==========================================

print("\n========== CONFUSION MATRIX ==========")

cm = confusion_matrix(y_test, y_pred)

print(cm)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoders["burnout_level"].classes_,
    yticklabels=label_encoders["burnout_level"].classes_
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig("graphs/confusion_matrix.png")

plt.close()

print("Confusion Matrix Graph Saved Successfully!")

# ==========================================
# Classification Report
# ==========================================

print("\n========== CLASSIFICATION REPORT ==========")

print(classification_report(y_test, y_pred))
# ==========================================
# Feature Importance
# ==========================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)

print("\n========== FEATURE IMPORTANCE ==========")
print(importance)

# ==========================================
# Feature Importance Graph
# ==========================================

plt.figure(figsize=(8,6))

importance_graph = importance.sort_values(
    by="Importance",
    ascending=True
)

plt.barh(
    importance_graph["Feature"],
    importance_graph["Importance"]
)

plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.tight_layout()

plt.savefig("graphs/feature_importance.png")

plt.close()

print("Feature Importance Graph Saved Successfully!")

# ==========================================
# Save Model
# ==========================================

joblib.dump(
    model,
    "model/burnout_model.pkl"
)

joblib.dump(
    label_encoders,
    "model/label_encoders.pkl"
)

print("\n========== MODEL SAVED ==========")

print("Model File   : model/burnout_model.pkl")
print("Encoder File : model/label_encoders.pkl")

print("\n==========================================")
print("Training Completed Successfully!")
print("==========================================")