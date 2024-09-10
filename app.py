import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBClassifier

# Load the saved Logistic Regression model
model = joblib.load('logR_model.pkl')

# Title and description for the app
st.title('Exoplanet Habitability Prediction')
st.write("This app predicts the habitability of exoplanets based on their features.")

# Sidebar for user input
st.sidebar.header('Customizable Features')

def user_input_features():
    s_age = st.sidebar.slider('Star Age (Billion Years)', 0.4, 8.0, 3.9)
    p_temp_surf = st.sidebar.slider('Surface Temperature (K)', 198.9, 325.4, 268.1)
    p_flux = st.sidebar.slider('Planetary Flux', 0.25, 1.64, 0.85)
    p_radius = st.sidebar.slider('Planet Radius (Earth Radii)', 0.79, 3.03, 1.72)
    p_mass = st.sidebar.slider('Planet Mass (Earth Mass)', 0.39, 36.0, 5.61)
    p_type = st.sidebar.selectbox('Planet Type', ('Jovian', 'Miniterran', 'Neptunian', 'Terran', 'Superterran', 'Subterran'))

    data = {
        'S_AGE': s_age,
        'P_TEMP_SURF': p_temp_surf,
        'P_FLUX': p_flux,
        'P_RADIUS': p_radius,
        'P_MASS': p_mass,
        'P_TYPE': p_type
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Apply one-hot encoding for P_TYPE to match training data
input_encoded = pd.get_dummies(input_df, columns=['P_TYPE'])

# Define the correct feature order (matching training data)
expected_columns = ['S_AGE', 'P_TEMP_SURF', 'P_FLUX', 'P_RADIUS', 'P_MASS',
                    'P_TYPE_Jovian', 'P_TYPE_Miniterran', 'P_TYPE_Neptunian', 'P_TYPE_Subterran', 'P_TYPE_Superterran', 'P_TYPE_Terran']

# Add missing columns with zeros and reorder the columns to match training
for col in expected_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0  # Add missing columns with value 0

# Reorder the columns to match the expected feature order
input_encoded = input_encoded[expected_columns]

# Display user inputs
st.subheader('User Input parameters')
st.write(input_df)

# Make prediction
prediction = model.predict(input_encoded)
prediction_proba = model.predict_proba(input_encoded)

# Display prediction
st.subheader('Prediction')
habitability = 'Habitable' if prediction[0] == 1 else 'Non-Habitable'
st.write(f"Predicted Habitability: **{habitability}**")

# Extract the probability for the positive class (1)
probability_positive = prediction_proba[0][1]

# Display probability for the positive class
st.subheader('Prediction Probability')
st.write(f"Probability of being Habitable: **{probability_positive:.2f}**")

import streamlit as st

# Footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #999;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        color: #f1f1f1;
    }
    </style>
    <div class="footer">
        Made by Alex T using data from https://phl.upr.edu/hwc
    </div>
    """,
    unsafe_allow_html=True
)
