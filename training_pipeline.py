import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import os

# Load the dataset
df = pd.read_csv('D:/financial_predictor/data/02. financial_dataset.csv')

# Debug: Print columns to verify
print("✅ Columns in dataset:", df.columns.tolist())

# Strip any whitespace from column names
df.columns = df.columns.str.strip()

# --- Data Preprocessing ---

# Fill missing values for numeric columns
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col].fillna(df[col].mean(), inplace=True)

# Fill missing values for categorical columns
for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Convert categorical columns to one-hot encoding
categorical_cols = ['Gender', 'Location', 'Customer_Segment']
existing_cats = [col for col in categorical_cols if col in df.columns]

if existing_cats:
    df = pd.get_dummies(df, columns=existing_cats, drop_first=True)

# --- Model Training ---

# Set correct target column
target_column = 'Future_Balance'

# Check if target column exists
if target_column not in df.columns:
    raise ValueError(f"The target column '{target_column}' is missing from the dataset. Found columns: {df.columns.tolist()}")

# Split data
X = df.drop(columns=target_column)
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
os.makedirs('saved_model', exist_ok=True)
with open('saved_model/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved successfully!")
