🔮 Future Financial Balance Predictor
This project is a web application built with Streamlit that predicts a user's future financial balance based on various personal and economic factors. It uses a pre-trained Linear Regression model to make its forecasts.

➡️ Click Here to View the Live Demo : https://financialbalancepredictor-l4wpnugcpa3ql7zqufrg9q.streamlit.app

✨ Features
Interactive UI: A user-friendly interface with sliders and input fields for all financial parameters.

Machine Learning Model: Utilizes a Linear Regression model trained on a financial dataset to provide predictions.

Instant Predictions: Get real-time forecasts for your future balance as you adjust the inputs.

Dual Currency Display: View the final prediction in both US Dollars ($) and Indian Rupees (₹).

Responsive Design: The application is fully functional on both desktop and mobile devices.

Log-Transformed Prediction: The model predicts the logarithm of the balance and converts it back, ensuring results are always positive and often more accurate.

🚀 How to Run Locally
To run this project on your own machine, follow these steps:

Clone the repository:

git clone https://github.com/tejasapkaje/financial_predictor.git
cd financial_predictor

Create and activate a virtual environment:

# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

Install the required libraries:

pip install -r requirements.txt

Run the Streamlit app:

streamlit run app.py

Your web browser will open with the local version of the application.

🛠️ Technology Stack
Language: Python

Web Framework: Streamlit

Machine Learning: Scikit-learn

Data Manipulation: Pandas, NumPy

📁 Project Structure
.
├── app.py                  # Main Streamlit application script
├── model.joblib            # Pre-trained machine learning model
├── scaler.joblib           # Pre-trained data scaler
├── requirements.txt        # Python libraries needed for the project
├── train_model.py          # Script to train the model from scratch
└── 02. financial_dataset.csv # The dataset used for training
