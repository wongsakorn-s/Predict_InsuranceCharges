import streamlit as st
import joblib
import pandas as pd

# Load the saved model
loaded_model = joblib.load('linear_regression_model.pkl')

# Define the features for input
features = ['age', 'bmi', 'children']

# Define the Streamlit app
st.title("Insurance Charges Prediction")

# Add input fields for user to enter data
age = st.slider("Age", min_value=18, max_value=64, value=30)
bmi = st.slider("BMI", min_value=15, max_value=50, value=25)
children = st.slider("Number of Children", min_value=0, max_value=5, value=2)

# Create a DataFrame with the user input
example_data = pd.DataFrame([[age, bmi, children]], columns=features)

# Make predictions
predicted_charges = loaded_model.predict(example_data)

# Display the prediction
st.write(f"Predicted Charges: ${predicted_charges[0]:.2f}")
