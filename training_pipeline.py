# train_model.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib

print("Starting model training process...")

# Load the dataset
df = pd.read_csv("C:\\Users\\SHREYAS\\Downloads\\TEJ-APK\\financial_balance_predictor\\data\\02. financial_dataset.csv")
print("Dataset loaded.")

# --- Preprocessing
num_cols = df.select_dtypes(include=['float64', 'int64']).columns.drop('Future_Balance')
for col in num_cols:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)
df['Customer_Segment'] = df['Customer_Segment'].fillna(df['Customer_Segment'].mode()[0])
for col in num_cols:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    df[col] = df[col].clip(lower, upper)
df = pd.get_dummies(df, columns=['Customer_Segment'], drop_first=True)
for col in df.columns:
    if df[col].dtype == 'bool':
        df[col] = df[col].astype(int)
print("Preprocessing complete.")

# Define features (X) and TRANSFORMED target (y)
X = df.drop('Future_Balance', axis=1)

y = np.log1p(df['Future_Balance'])

# Initialize and fit the scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Scaler fitted.")

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_scaled, y)
print("Model trained on log-transformed target.")

# Save the column order for prediction
model.columns = X.columns

# Save the model and scaler
joblib.dump(model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("Model and scaler have been saved.")