import streamlit as st
import pandas as pd
import joblib

# Load the trained model and expected column order
model = joblib.load('model.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("Customer Churn Predictor")
st.write("Enter a customer's details to predict their likelihood of churning.")

# --- Input fields ---
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=float(tenure * monthly_charges))

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.selectbox("Payment Method", 
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

gender = st.selectbox("Gender", ["Female", "Male"])
senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Has Partner", ["No", "Yes"])
dependents = st.selectbox("Has Dependents", ["No", "Yes"])
phone_service = st.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
online_security = st.selectbox("Online Security", ["No", "Yes"])
online_backup = st.selectbox("Online Backup", ["No", "Yes"])
device_protection = st.selectbox("Device Protection", ["No", "Yes"])
tech_support = st.selectbox("Tech Support", ["No", "Yes"])
streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])

# --- Build a single-row DataFrame matching the model's expected input ---
input_dict = {
    'gender': 1 if gender == 'Male' else 0,
    'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
    'Partner': 1 if partner == 'Yes' else 0,
    'Dependents': 1 if dependents == 'Yes' else 0,
    'tenure': tenure,
    'PhoneService': 1 if phone_service == 'Yes' else 0,
    'MultipleLines': 1 if multiple_lines == 'Yes' else 0,
    'OnlineSecurity': 1 if online_security == 'Yes' else 0,
    'OnlineBackup': 1 if online_backup == 'Yes' else 0,
    'DeviceProtection': 1 if device_protection == 'Yes' else 0,
    'TechSupport': 1 if tech_support == 'Yes' else 0,
    'StreamingTV': 1 if streaming_tv == 'Yes' else 0,
    'StreamingMovies': 1 if streaming_movies == 'Yes' else 0,
    'PaperlessBilling': 1 if paperless_billing == 'Yes' else 0,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'InternetService_Fiber optic': 1 if internet_service == 'Fiber optic' else 0,
    'InternetService_No': 1 if internet_service == 'No' else 0,
    'Contract_One year': 1 if contract == 'One year' else 0,
    'Contract_Two year': 1 if contract == 'Two year' else 0,
    'PaymentMethod_Credit card (automatic)': 1 if payment_method == 'Credit card (automatic)' else 0,
    'PaymentMethod_Electronic check': 1 if payment_method == 'Electronic check' else 0,
    'PaymentMethod_Mailed check': 1 if payment_method == 'Mailed check' else 0,
}

input_df = pd.DataFrame([input_dict])
input_df = input_df[model_columns]  # enforce exact column order the model expects

# --- Predict ---
if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High churn risk — predicted probability: {probability:.1%}")
    else:
        st.success(f"✅ Low churn risk — predicted probability: {probability:.1%}")