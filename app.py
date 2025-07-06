import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('saved_model/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Set Streamlit page config
st.set_page_config(page_title="💸 Financial Spend Predictor", layout="wide")

# Title and Description
st.markdown("<h1 style='text-align: center;'>💸 Financial Spend Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Predict your total spending in INR based on key financial inputs 🧾</h4>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar - User Input
with st.sidebar:
    st.image("https://img.icons8.com/ios/452/money--v1.png", width=100)
    st.header("🧍‍♂️ User Input Features")

    age = st.number_input("🎂 Age", 18, 100, 30)
    income = st.number_input("💼 Annual Income (in ₹ thousands)", 100, 10000, 5000)
    credit_score = st.number_input("🏦 Credit Score", 300, 850, 650)
    gender = st.selectbox("👫 Gender", ("Male", "Female"))
    location = st.selectbox("📍 Location", ("Rural", "Urban", "Suburban"))

    predict_btn = st.button("🔮 Predict Total Spend")

# Convert categorical inputs
gender_male = 1 if gender == "Male" else 0
location_suburban = 1 if location == "Suburban" else 0
location_urban = 1 if location == "Urban" else 0

# Prediction Section
if predict_btn:
    st.subheader("📊 Prediction Result")

    input_data = pd.DataFrame({
        'Age': [age],
        'Annual_Income_kINR': [income],
        'Credit_Score': [credit_score],
        'Gender_Male': [gender_male],
        'Location_Suburban': [location_suburban],
        'Location_Urban': [location_urban]
    })

    # Ensure all expected columns exist
    model_cols = model.feature_names_in_
    input_data = input_data.reindex(columns=model_cols, fill_value=0)

    prediction = model.predict(input_data)[0]

    st.success(f"💰 **Estimated Total Spend**: ₹{prediction:,.2f}")

    # Optional: Show input summary
    with st.expander("📄 See your input summary"):
        st.table(input_data.T)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center;'>"
    "Built with ❤️ using Streamlit | Powered by ML"
    "</div>",
    unsafe_allow_html=True
)
