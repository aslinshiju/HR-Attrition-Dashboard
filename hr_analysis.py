import matplotlib
matplotlib.use('Agg')  #  non-GUI backend

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

print("\n Script started...")

# Step 1: Load data from CSV
try:
    df = pd.read_csv("../data/HR_Employee_Attrition.csv")
    print(f"\n CSV Loaded - Rows: {df.shape[0]} , Columns: {df.shape[1]}")
except Exception as e:
    print(" ERROR loading CSV:", e)
    exit()

# Step 2: Explore Columns
print("\n Columns:", df.columns.tolist())
print("\n Missing Values:\n", df.isnull().sum())

# Step 3: Create outputs/ folder for plots
os.makedirs('outputs', exist_ok=True)

# Step 4: EDA Plots
try:
    sns.countplot(data=df, x='Attrition', palette='Set2', hue='Attrition', legend=False)
    plt.title("Attrition Count")
    plt.savefig("outputs/plot_attrition_count.png")
    plt.close()

    sns.boxplot(x='Attrition', y='Age', data=df, palette='coolwarm')
    plt.title("Attrition vs Age")
    plt.savefig("outputs/plot_age_vs_attrition.png")
    plt.close()

    print(" All EDA visualizations completed and saved to outputs/")
except Exception as e:
    print(" Error saving plots:", e)

# Step 5: Clean Data
try:
    df.drop(['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber'], axis=1, inplace=True)
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    df = pd.get_dummies(df, drop_first=True)
    print("\n🧹 Data cleaned successfully")
except Exception as e:
    print(" ERROR cleaning data:", e)
    exit()

# Step 6: Train ML Model
try:
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n Model trained")
    print("\n Accuracy:", model.score(X_test, y_test))
    print("\n Classification Report:\n", classification_report(y_test, y_pred))
    print(f"\n X_test shape: {X_test.shape}")
    print(f" Sample Predictions: {y_pred[:5]}")
except Exception as e:
    print(" ERROR training model:", e)
    exit()

# Step 7: Export full info + prediction to Excel
try:
    # Reload original dataset to get readable info
    df_original = pd.read_csv("../data/HR_Employee_Attrition.csv")

    # Add RowID for safe merge
    df_original['RowID'] = df_original.index
    X_test['RowID'] = X_test.index

    # Add prediction and actual
    predictions = X_test.copy()
    predictions['Predicted'] = y_pred
    predictions['Actual'] = y_test.values
    predictions['RowID'] = predictions.index

    # Merge with original (to get readable columns like Department, Gender)
    merged = pd.merge(predictions, df_original, on='RowID')

    # Export
    os.makedirs('excel', exist_ok=True)
    merged.to_excel("excel/model_predictions_merged.xlsx", index=False)
    print("\n Excel export completed: excel/model_predictions_merged.xlsx")
except Exception as e:
    print(" ERROR exporting merged Excel:", e)
