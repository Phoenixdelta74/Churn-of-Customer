import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder


# Load the trained model, scaler, and encoders
model = load_model('model.h5')

with open('geo_encoded_value.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('Label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('Scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


## Streamlit app
st.set_page_config(page_title='Customer Churn Prediction', page_icon='📊')
st.title('📊 Customer Churn Prediction')

st.markdown("""
This application uses an **Artificial Neural Network (ANN)** to predict the probability of a customer leaving the bank (churn).
Please enter the customer details below:
""")

# User input in columns for better layout
col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('Gender', label_encoder_gender.classes_)
    age = st.slider('Age', 18, 92, 35)
    balance = st.number_input('Account Balance', min_value=0.0, step=1000.0)
    credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=600)

with col2:
    estimated_salary = st.number_input('Estimated Salary', min_value=0.0, step=1000.0)
    tenure = st.slider('Tenure (Years)', 0, 10, 5)
    num_of_products = st.slider('Number of Products', 1, 4, 1)
    has_cr_card = st.selectbox('Has Credit Card?', ['No', 'Yes'])
    is_active_member = st.selectbox('Is Active Member?', ['No', 'Yes'])

# Map 'Yes/No' to 1/0
has_cr_card = 1 if has_cr_card == 'Yes' else 0
is_active_member = 1 if is_active_member == 'Yes' else 0

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})


# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)


## Scaling the data
input_scaled = scaler.transform(input_data)

## Prediction 
prediction = model.predict(input_scaled)
prediction_proba = prediction[0][0]

st.divider()
st.subheader('Prediction Result')
st.write(f'**Churn Probability:** `{prediction_proba:.2%}`')

if prediction_proba > 0.5:
    st.error("⚠️ The customer is **likely to churn**.")
else:
    st.success("✅ The customer is **not likely to churn**.")
