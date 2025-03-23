import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
raw_data = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

# Load model, encoder, and scaler with joblib
model = joblib.load('model.pkl')
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')

# Title and description
st.title('Obesity Prediction ML App')
st.info('This app predicts your obesity level based on your inputs!')

# =================== Display Raw Data ===================
with st.expander('Show Raw Data'):
    st.dataframe(raw_data, use_container_width=True)

# =================== Data Visualization ===================
with st.expander('Data Visualization'):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=raw_data,
        x='Height',
        y='Weight',
        hue='NObeyesdad',
        palette='Set1'
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    st.pyplot(fig)

# =================== User Inputs ===================
st.subheader('Input Your Data')

gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', 10, 100, 25)
height = st.number_input('Height (m)', min_value=1.0, max_value=2.5, value=1.7)
weight = st.number_input('Weight (kg)', min_value=30.0, max_value=200.0, value=70.0)
family_history = st.selectbox('Family History with Overweight', ['yes', 'no'])
favc = st.selectbox('Frequent High Calorie Food Consumption', ['yes', 'no'])
fcvc = st.slider('Vegetable Consumption Frequency (1-3)', 1, 3, 2)
ncp = st.slider('Number of Main Meals', 1, 4, 3)
caec = st.selectbox('Food between Meals', ['no', 'Sometimes', 'Frequently', 'Always'])
smoke = st.selectbox('Do you Smoke?', ['yes', 'no'])
ch2o = st.slider('Water Consumption (1-3)', 1, 3, 2)
scc = st.selectbox('Do you Monitor your Calories?', ['yes', 'no'])
faf = st.slider('Physical Activity Frequency (0-3)', 0, 3, 1)
tue = st.slider('Time Using Technology (0-3)', 0, 3, 1)
calc = st.selectbox('Alcohol Consumption', ['no', 'Sometimes', 'Frequently', 'Always'])
mtrans = st.selectbox('Transportation Mode', ['Automobile', 'Bike', 'Motorbike', 'Public Transportation', 'Walking'])

# Create DataFrame for user input
input_data = pd.DataFrame({
    'Gender': [gender],
    'Age': [age],
    'Height': [height],
    'Weight': [weight],
    'family_history_with_overweight': [family_history],
    'FAVC': [favc],
    'FCVC': [fcvc],
    'NCP': [ncp],
    'CAEC': [caec],
    'SMOKE': [smoke],
    'CH2O': [ch2o],
    'SCC': [scc],
    'FAF': [faf],
    'TUE': [tue],
    'CALC': [calc],
    'MTRANS': [mtrans]
})

st.subheader('Your Input Data')
st.dataframe(input_data, use_container_width=True)

# =================== Encoding & Scaling ===================
if st.button('Predict Obesity Level'):
    try:
        # Transform categorical features using LabelEncoder
        for col in encoder.keys():
            input_data[col] = encoder[col].transform(input_data[[col]])

        # Scale only age, weight, ncp
        scale_cols = ['Age', 'Weight', 'NCP']
        input_data[scale_cols] = scaler.transform(input_data[scale_cols])

        # Predict
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)

        st.success(f'Predicted Obesity Level: {prediction[0]}')
        st.info(f'Prediction Probability: {np.max(probability) * 100:.2f}%')

    except Exception as e:
        st.error(f'Prediction failed: {e}')
