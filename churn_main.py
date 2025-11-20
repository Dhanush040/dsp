import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from joblib import dump

# Load CSV (must be in the same folder)
df = pd.read_csv("Telco_Customer_Churn.csv")

# Clean TotalCharges
df['TotalCharges'] = pd.to_numeric(df.get('TotalCharges', pd.Series()), errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df.get('tenure', 0) * df.get('MonthlyCharges', 0))
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Map target
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# Features (include gender and PaymentMethod)
selected_features = [
    'tenure',
    'InternetService',
    'Contract',
    'MonthlyCharges',
    'TotalCharges',
    'gender',
    'PaymentMethod'
]

missing = [c for c in selected_features if c not in df.columns]
if missing:
    raise SystemExit(f"Missing columns in CSV: {missing}")

X = df[selected_features].copy()
y = df['Churn'].copy()

# numeric / categorical split
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
cat_cols = [c for c in selected_features if c not in numeric_cols]

# Encode categorical columns (LabelEncoder per column) and save mapping
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"Mapping for {col}: {encoders[col]}")

# Scale numeric
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Train/test split (optional)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101, stratify=y)

# Train logistic regression
model = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=101)
model.fit(X_train, y_train)

# Save artifacts using the EXACT filenames your Streamlit app checks
dump(model, 'logistic_regression_model.joblib')
dump(encoders, 'encoders_gender_payment.joblib')
dump(selected_features, 'selected_features.joblib')
dump(scaler, 'scaler.joblib')  # optional, helpful for inference

print("Saved: logistic_regression_model.joblib, encoders_gender_payment.joblib, selected_features.joblib, scaler.joblib")

