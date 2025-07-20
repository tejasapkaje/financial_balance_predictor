import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. SETUP AND STYLING
st.set_page_config(
    page_title="Financial Future Predictor",
    page_icon="🔮",
    layout="wide"
)

# Configuration
USD_TO_INR_RATE = 83.50  # Using a fixed rate for conversion

# Enhanced Styling with CSS
st.markdown(f"""
<style>
    /* Main app background and font */
    .stApp {{
        background: #f0f8ff; /* Light Alice Blue background */
        font-family: 'Helvetica', sans-serif;
    }}

    /* Main title styling */
    .title-container {{
        text-align: center;
        padding: 2rem;
        background-image: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
        border-radius: 12px;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }}
    .title-container h1 {{
        font-size: 3rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }}

    /* Sidebar styling */
    .css-1d391kg {{
        background-color: #ffffff;
        border-right: 2px solid #e0e0e0;
    }}
    .css-1d391kg h2 {{
        color: #1e3a8a;
    }}

    /* Button styling */
    .stButton>button {{
        background-image: linear-gradient(to right, #2563eb, #1d4ed8);
        color: white;
        border-radius: 12px;
        padding: 12px 28px;
        border: none;
        font-size: 20px;
        font-weight: bold;
        transition: all 0.3s;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        width: 100%;
    }}
    .stButton>button:hover {{
        transform: scale(1.05);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
    }}

    /* Result display styling */
    .result-box {{
        background-color: #ffffff;
        border: 2px solid #2563eb;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }}
    .result-box h3 {{
        color: #1e3a8a;
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }}
    .result-box p {{
        color: #2563eb;
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
    }}
</style>
""", unsafe_allow_html=True)


# 2. LOAD THE PRE-TRAINED MODEL AND SCALER
@st.cache_resource
def load_model():
    """Loads the pre-trained model and scaler from disk."""
    try:
        model = joblib.load('model.joblib')
        scaler = joblib.load('scaler.joblib')
        return model, scaler
    except FileNotFoundError:
        st.error("Model/scaler not found. Ensure 'model.joblib' & 'scaler.joblib' are in the repository.")
        return None, None

model, scaler = load_model()


# 3. DEFINE THE USER INPUT INTERFACE WITH EMOJIS
def user_input_features():
    """Creates sidebar widgets with emojis and returns a DataFrame of user inputs."""
    st.sidebar.title("👤 Your Financial Profile")
    st.sidebar.markdown("---")

    age = st.sidebar.slider('🎂 Age', 21, 65, 40)
    annual_income = st.sidebar.number_input('💰 Annual Income ($)', min_value=300000, max_value=5000000, value=900000, step=10000)
    monthly_expenses = st.sidebar.number_input('💸 Monthly Expenses ($)', min_value=10000, max_value=200000, value=40000, step=1000)
    savings_rate = st.sidebar.slider('📈 Savings Rate (%)', 0.0, 1.0, 0.25, 0.01)
    debt_to_income_ratio = st.sidebar.slider('📉 Debt to Income Ratio', 0.0, 2.0, 0.3, 0.01)
    current_investments = st.sidebar.number_input('🏦 Current Investments ($)', min_value=100000, max_value=10000000, value=2000000, step=10000)
    total_loan_amount = st.sidebar.number_input('💳 Total Loan Amount ($)', min_value=0, max_value=5000000, value=500000, step=10000)
    avg_credit_score = st.sidebar.slider('📊 Average Credit Score', 300, 850, 670)
    inflation_rate = st.sidebar.slider('🔥 Assumed Inflation Rate (%)', 2.0, 10.0, 6.0, 0.1)
    interest_rate = st.sidebar.slider('💹 Assumed Interest Rate (%)', 3.0, 15.0, 7.0, 0.1)
    years_of_employment = st.sidebar.slider('🧑‍💼 Years of Employment', 0.0, 40.0, 8.0, 0.5)
    job_stability_score = st.sidebar.slider('⚖️ Job Stability Score', 0.0, 1.0, 0.5, 0.01)
    emergency_fund_value = st.sidebar.number_input('🛡️ Emergency Fund ($)', min_value=0, max_value=1000000, value=200000, step=5000)
    retirement_fund_contribution = st.sidebar.number_input('🏖️ Retirement Fund Contrib. ($)', min_value=0, max_value=500000, value=90000, step=1000)
    customer_segment = st.sidebar.selectbox('🧩 Customer Segment', ('Bronze', 'Silver', 'Gold'))

    data = {
        'Age': age, 'Annual_Income': annual_income, 'Monthly_Expenses': monthly_expenses,
        'Savings_Rate': savings_rate, 'Debt_to_Income_Ratio': debt_to_income_ratio,
        'Current_Investments_Value': current_investments, 'Total_Loan_Amount': total_loan_amount,
        'Avg_Credit_Score': avg_credit_score, 'Inflation_Rate': inflation_rate,
        'Interest_Rate': interest_rate, 'Years_of_Employment': years_of_employment,
        'Job_Stability_Score': job_stability_score, 'Emergency_Fund_Value': emergency_fund_value,
        'Retirement_Fund_Contribution': retirement_fund_contribution,
        'Customer_Segment_Gold': 1 if customer_segment == 'Gold' else 0,
        'Customer_Segment_Silver': 1 if customer_segment == 'Silver' else 0,
    }
    features = pd.DataFrame(data, index=[0])
    return features


# 4. DISPLAY THE UI AND PREDICTION LOGIC
st.markdown("<div class='title-container'><h1>Future Financial Balance Predictor</h1></div>", unsafe_allow_html=True)
st.markdown("##### Welcome! This tool uses a machine learning model to forecast your potential financial balance. Fill in your details in the sidebar to get started.")
#  # This is just a placeholder to show where an image could go.

# Get user inputs
input_df = user_input_features()

col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.markdown("### Your Current Financial Snapshot")
    st.dataframe(input_df.T.rename(columns={0: 'Your Inputs'}).style.format("{:,.2f}"))

with col2:
    st.markdown("### Ready to See Your Future?")
    st.markdown("Once you've adjusted the sliders and inputs on the left, click the button below to generate your personalized prediction.")
    
    if st.button('🔮 Predict My Future Balance'):
        if model is not None and scaler is not None:
            # Align columns and scale
            input_df_aligned = input_df.reindex(columns=model.columns, fill_value=0)
            input_scaled = scaler.transform(input_df_aligned)

            # Make prediction
            log_prediction = model.predict(input_scaled)
            final_prediction_usd = np.expm1(log_prediction)
            
            # CURRENCY CONVERSION
            final_prediction_inr = final_prediction_usd * USD_TO_INR_RATE

            # Display results in two columns
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.markdown(f"""
                <div class="result-box">
                    <h3>Prediction (in USD)</h3>
                    <p>${final_prediction_usd[0]:,.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            with res_col2:
                st.markdown(f"""
                <div class="result-box">
                    <h3>Prediction (in INR)</h3>
                    <p>₹{final_prediction_inr[0]:,.2f}</p>
                </div>
                """, unsafe_allow_html=True)
                st.caption(f"Based on a conversion rate of 1 USD = {USD_TO_INR_RATE} INR.")
        else:
            st.error("Model is not loaded. Please check your setup.")

st.markdown("---")
st.info("Disclaimer: This prediction is based on a statistical model and is for informational purposes only. It is not financial advice.")