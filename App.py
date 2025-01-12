import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, Normalizer


try:
    model = joblib.load('Loan_prediction_model_AutoML.pkl')
except FileNotFoundError:
    print("Model file not found. Train or define the model first.")


def preprocess_and_predict(data):
    # Feature engineering
    data= pd.concat([data, pd.get_dummies(data['person_home_ownership'], prefix='person_home_ownership')], axis=1)
    data= pd.concat([data, pd.get_dummies(data['loan_intent'], prefix='loan_intent')], axis=1)

    data['cb_person_default_on_file'] = data['cb_person_default_on_file'].replace({'N': 0, 'Y': 1})

    label_encoder = LabelEncoder()
    data['loan_grade'] = label_encoder.fit_transform(data['loan_grade'])

    # Select relevant features for prediction
    columns=[ 'person_age', 'person_income',
       'person_emp_length', 'loan_grade', 'loan_amnt',
       'loan_int_rate', 'loan_percent_income', 'cb_person_default_on_file',
       'cb_person_cred_hist_length',
       'person_home_ownership_MORTGAGE', 'person_home_ownership_OWN',
       'person_home_ownership_RENT', 'loan_intent_DEBTCONSOLIDATION',
       'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT',
       'loan_intent_MEDICAL', 'loan_intent_PERSONAL', 'loan_intent_VENTURE']

    # Ensure all required columns are present in the DataFrame
    for column in columns:
        if column not in data.columns:
            data[column] = 0  # Set missing column to zero

    features=data[columns]


    features_Numeric = features[['person_age', 'person_income','person_emp_length','loan_amnt','loan_int_rate','loan_percent_income','cb_person_cred_hist_length']]

    # Scaling
    scaler_normalizer = Normalizer()
    features_Normalized_NotEncoded = scaler_normalizer.fit_transform(features_Numeric)
    features_Normalized_NotEncoded = pd.DataFrame(features_Normalized_NotEncoded, columns=features_Numeric.columns)

    features.reset_index(drop=True, inplace=True)
    features_Normalized_NotEncoded.reset_index(drop=True, inplace=True)

    scaled_features = pd.concat([features_Normalized_NotEncoded, features[['loan_grade','cb_person_default_on_file',
       'person_home_ownership_MORTGAGE', 'person_home_ownership_OWN',
       'person_home_ownership_RENT', 'loan_intent_DEBTCONSOLIDATION',
       'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT',
       'loan_intent_MEDICAL', 'loan_intent_PERSONAL', 'loan_intent_VENTURE']]], axis=1)
    
    # Predict cluster
    predictions = model.predict(scaled_features)
    data['loan_status'] = predictions
    return data



st.title("Loan approval Predictor")
st.write("This app predict if your loan request is going to be approved.")

# User input
st.sidebar.header("Input Loan Data")


# Create form for user input
person_age = st.sidebar.slider("person age", 18, 75, 30)
person_income = st.sidebar.number_input("person income", min_value=0, step=500)
person_home_ownership = st.sidebar.selectbox("person home ownership", ["RENT", "MORTGAGE", "OWN"])
person_emp_length = st.sidebar.number_input("person emp length", min_value=0, step=2)
loan_intent = st.sidebar.selectbox("loan intent", ["EDUCATION", "MEDICAL", "PERSONAL", "VENTURE", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
loan_grade = st.sidebar.selectbox("loan grade", ["A", "B", "C", "D", "E", "F", "G"])
loan_amnt = st.sidebar.number_input("loan amount", min_value=0, step=500)
loan_int_rate = st.sidebar.number_input("loan int rate", min_value=0.0, step=0.01)
cb_person_default_on_file = st.sidebar.selectbox("cb person default on file", ["Y", "N"])
cb_person_cred_hist_length = st.sidebar.number_input("cb person cred hist length", min_value=0, step=1)


# Predict button
if st.sidebar.button("Predict approval"):

    input_df = pd.DataFrame({
        'person_age': [person_age],
        'person_income': [person_income],
        'person_home_ownership': [person_home_ownership],
        'person_emp_length': [person_emp_length],
        'loan_intent': [loan_intent],
        'loan_grade': [loan_grade],
        'loan_amnt': [loan_amnt],
        'loan_int_rate': [loan_int_rate],
        'cb_person_default_on_file': [cb_person_default_on_file],
        'cb_person_cred_hist_length': [cb_person_cred_hist_length]
    })

    # Make prediction
    result_df = preprocess_and_predict(input_df)
    
    # Display results
    st.subheader("loan_status")
    st.write(result_df)

    # Interpret the Status
    status = result_df['loan_status'].values[0]
    if status >= 0.5:
        st.write("The loan is approved.")
    elif status <= 0.5:
        st.write("The loan is not approved.")
