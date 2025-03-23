import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Load raw dataset for display and visualization
raw_data = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

# Load pkl files
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

# ========================== 3. User Inputs ===========================
st.header('Input Data')
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

# Create DataFrame from user input
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

# ========================= 4. Encoding ===============================
categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 
                    'SMOKE', 'SCC', 'CALC', 'MTRANS']
input_categorical = input_data[categorical_cols]
encoded_categorical = encoder.transform(input_categorical)
encoded_categorical_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_cols))

# ========================= 5. Scaling (only Age, Weight, NCP) ====================
scale_cols = ['Age', 'Weight', 'NCP']
input_numerical = input_data[scale_cols]
scaled_numerical = scaler.transform(input_numerical)
scaled_numerical_df = pd.DataFrame(scaled_numerical, columns=scale_cols)

# Get the other features (without scaling)
other_cols = input_data.drop(columns=categorical_cols + scale_cols).reset_index(drop=True)

# ========================= 6. Concatenate all processed features =================
final_input = pd.concat([scaled_numerical_df, other_cols, encoded_categorical_df], axis=1)

# ========================= 7. Prediction Button ============================
if st.button('Predict'):
    prediction = model.predict(final_input)
    probability = model.predict_proba(final_input)

    st.success(f'Prediction: {prediction[0]}')
    st.info(f'Prediction Probability: {np.max(probability) * 100:.2f}%')
