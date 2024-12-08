import streamlit as st
import pandas as pd
import pickle

# Load pre-trained model and encoders
def load_model_and_encoders():
    with open('customer_churn_model.pkl', 'rb') as model_file:
        model_data = pickle.load(model_file)
        model = model_data["model"]  # Access the 'model' from the dictionary
    with open('encoders.pkl', 'rb') as encoders_file:
        encoders = pickle.load(encoders_file)
    return model, encoders

# Preprocess user input to ensure all necessary columns are present
def preprocess_input(input_data, encoders, feature_columns):
    # Add missing columns with default values
    for col in feature_columns:
        if col not in input_data:
            input_data[col] = "No"  # Default value for missing categorical features
        # Encode categorical variables
        if col in input_data:
            encoder = encoders.get(col)
            if encoder:
                input_data[col] = encoder.transform([input_data[col]])[0]
    # Ensure columns are in the same order as the training data
    return [input_data.get(col) for col in feature_columns]

# Streamlit app layout
def main():
    st.title("Customer Churn Prediction App")
    st.write("Predict whether a customer will churn based on their data.")

    # User input form
    st.sidebar.header("Customer Details")
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
    partner = st.sidebar.selectbox("Has a Partner?", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Has Dependents?", ["Yes", "No"])
    tenure = st.sidebar.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
    phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.sidebar.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    monthly_charges = st.sidebar.slider("Monthly Charges", min_value=0, max_value=150, value=50)
    total_charges = st.sidebar.slider("Total Charges", min_value=0, max_value=10000, value=1000)

    # Prepare input data
    input_data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "InternetService": internet_service,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    # Load the model and encoders
    model, encoders = load_model_and_encoders()

    # Load feature columns (from training process)
    feature_columns = pd.read_pickle('customer_churn_model.pkl')["features_names"]

    # Preprocess the input data
    processed_data = preprocess_input(input_data, encoders, feature_columns)

    # Convert the input to DataFrame
    input_df = pd.DataFrame([processed_data], columns=feature_columns)

    # Predict
    if st.button("Predict"):
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        result = "Churn" if prediction == 1 else "No Churn"
        st.write(f"Prediction: **{result}**")
        st.write(f"Churn Probability: **{probability:.2f}**")

if __name__ == "__main__":
    main()
