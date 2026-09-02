# ==========================================
# Employee Burnout Prediction
# Exploratory Data Analysis (EDA)
# ==========================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# Create Graph Folder
# ==========================================

os.makedirs("graphs", exist_ok=True)

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("dataset/employee_burnout_synthetic_50000.csv")

print("\n========== DATASET LOADED ==========")
print(df.head())

print("\nDataset Shape :", df.shape)

print("\n========== DATASET INFO ==========")
print(df.info())

print("\n========== COLUMNS ==========")
print(df.columns)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE VALUES ==========")
print(df.duplicated().sum())

# ==========================================
# Graph Style
# ==========================================

sns.set_style("whitegrid")
# ==========================================
# 1. Burnout Level Distribution
# ==========================================

plt.figure(figsize=(6,4))

sns.countplot(
    x="burnout_level",
    data=df,
    palette="Set2"
)

plt.title("Burnout Level Distribution")
plt.xlabel("Burnout Level")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig(
    "graphs/burnout_distribution.png",
    dpi=300
)

plt.close()

# ==========================================
# 2. Gender Distribution
# ==========================================

plt.figure(figsize=(6,4))

sns.countplot(
    x="gender",
    data=df,
    palette="Pastel1"
)

plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig(
    "graphs/gender_distribution.png",
    dpi=300
)

plt.close()

# ==========================================
# 3. Job Role Distribution
# ==========================================

plt.figure(figsize=(10,5))

sns.countplot(
    y="job_role",
    data=df,
    order=df["job_role"].value_counts().index,
    palette="viridis"
)

plt.title("Job Role Distribution")
plt.xlabel("Count")
plt.ylabel("Job Role")

plt.tight_layout()

plt.savefig(
    "graphs/job_role_distribution.png",
    dpi=300
)

plt.close()

# ==========================================
# 4. Work Hours vs Burnout
# ==========================================

plt.figure(figsize=(8,5))

sns.boxplot(
    x="burnout_level",
    y="work_hours_per_week",
    data=df,
    palette="Set3"
)

plt.title("Work Hours vs Burnout Level")
plt.xlabel("Burnout Level")
plt.ylabel("Work Hours Per Week")

plt.tight_layout()

plt.savefig(
    "graphs/workhours_vs_burnout.png",
    dpi=300
)

plt.close()

# ==========================================
# 5. Sleep Hours vs Burnout
# ==========================================

plt.figure(figsize=(8,5))

sns.boxplot(
    x="burnout_level",
    y="sleep_hours",
    data=df,
    palette="Set3"
)

plt.title("Sleep Hours vs Burnout Level")
plt.xlabel("Burnout Level")
plt.ylabel("Sleep Hours")

plt.tight_layout()

plt.savefig(
    "graphs/sleep_vs_burnout.png",
    dpi=300
)

plt.close()
# ==========================================
# 6. Work-Life Balance vs Burnout
# ==========================================

plt.figure(figsize=(8,5))

sns.boxplot(
    x="burnout_level",
    y="work_life_balance",
    data=df,
    palette="Set3"
)

plt.title("Work-Life Balance vs Burnout")
plt.xlabel("Burnout Level")
plt.ylabel("Work-Life Balance")

plt.tight_layout()

plt.savefig(
    "graphs/worklife_balance.png",
    dpi=300
)

plt.close()

# ==========================================
# 7. Job Satisfaction vs Burnout
# ==========================================

plt.figure(figsize=(8,5))

sns.boxplot(
    x="burnout_level",
    y="job_satisfaction",
    data=df,
    palette="Set3"
)

plt.title("Job Satisfaction vs Burnout")
plt.xlabel("Burnout Level")
plt.ylabel("Job Satisfaction")

plt.tight_layout()

plt.savefig(
    "graphs/job_satisfaction.png",
    dpi=300
)

plt.close()

# ==========================================
# 8. Manager Support vs Burnout
# ==========================================

plt.figure(figsize=(8,5))

sns.boxplot(
    x="burnout_level",
    y="manager_support",
    data=df,
    palette="Set3"
)

plt.title("Manager Support vs Burnout")
plt.xlabel("Burnout Level")
plt.ylabel("Manager Support")

plt.tight_layout()

plt.savefig(
    "graphs/manager_support.png",
    dpi=300
)

plt.close()

# ==========================================
# 9. Correlation Heatmap
# ==========================================

numeric_df = df.select_dtypes(include=["int64", "float64"])

plt.figure(figsize=(8,6))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    square=True
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "graphs/correlation_heatmap.png",
    dpi=300
)

plt.close()

# ==========================================
# 10. Age Distribution
# ==========================================

plt.figure(figsize=(8,5))

sns.histplot(
    df["age"],
    bins=20,
    kde=True,
    color="steelblue"
)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig(
    "graphs/age_distribution.png",
    dpi=300
)

plt.close()

# ==========================================
# EDA Completed
# ==========================================

print("\n===================================")
print("EDA Completed Successfully!")
print("All Graphs Saved in 'graphs' Folder.")
print("===================================")