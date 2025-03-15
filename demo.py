import streamlit as st
import numpy as np
from winequality.pipeline.prediction import PredictionPipeline
import os

def train_model():
    os.system("python main.py")
    st.success("Training Successful!")

# Streamlit UI
st.title("Wine Quality Prediction App")

# Sidebar navigation
option = st.sidebar.radio("Navigation", ["Home", "Train Model", "Predict"])

if option == "Home":
    st.write("Welcome to the Wine Quality Prediction App!")
    st.write("Use the sidebar to navigate.")

elif option == "Train Model":
    if st.button("Train Now"):
        train_model()

elif option == "Predict":
    st.subheader("Enter Wine Features")
    
    fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, format="%.2f")
    volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, format="%.2f")
    citric_acid = st.number_input("Citric Acid", min_value=0.0, format="%.2f")
    residual_sugar = st.number_input("Residual Sugar", min_value=0.0, format="%.2f")
    chlorides = st.number_input("Chlorides", min_value=0.0, format="%.4f")
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, format="%.2f")
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, format="%.2f")
    density = st.number_input("Density", min_value=0.0, format="%.4f")
    pH = st.number_input("pH", min_value=0.0, format="%.2f")
    sulphates = st.number_input("Sulphates", min_value=0.0, format="%.2f")
    alcohol = st.number_input("Alcohol", min_value=0.0, format="%.2f")

    if st.button("Predict Quality"):
        try:
            data = np.array([
                fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
                pH, sulphates, alcohol
            ]).reshape(1, -1)
            
            obj = PredictionPipeline()
            prediction = obj.predict(data)
            st.success(f"Predicted Wine Quality: {prediction}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
