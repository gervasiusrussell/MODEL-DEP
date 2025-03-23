import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Load your raw dataset for display and visualization
raw_data = pd.read_csv('your_dataset.csv')  # Replace with your dataset file

# Load the pickle files
encoder = pickle.load(open('encoder.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# App Title
st.title('Machine Learning App')
st.info('This app will predict your obesity level!')

# ======================= 1. Display Raw Data ========================
with st.expander('Data'):
    st.write('This is a raw data')
    st.dataframe(raw_data)

# ======================= 2. Data Visualization ======================
with st.expander('Data Visualization'):
    st.write('Data Visualization')
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=raw_data,
        x='Height',
        y='Weight',
        hue='NObeyesdad',
        palette='Set1'
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    st.pyplot(fig)

# User Inputs
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', 10, 100, 25)
height = st.number_input('Height (in meters)', min_value=1.0, max_value=2.5, value=1.7)
weight = st.number_input('Weight (in kg)', min_value=30.0, max_value=200.0, value=70.0)
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

# Create DataFrame
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

# Encoding
input_encoded = encoder.transform(input_data)

# Normalization/Scaling
input_scaled = scaler.transform(input_encoded)

# Prediction
if st.button('Predict'):
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    st.success(f'Prediction: {prediction[0]}')
    st.info(f'Prediction Probability: {np.max(probability) * 100:.2f}%')
