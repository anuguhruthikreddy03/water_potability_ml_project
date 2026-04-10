import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Water Potability Predictor",
    layout="centered",
    page_icon="💧"
)

@st.cache_resource
def load_model():
    try:
        model = joblib.load("water_potability_pipeline.pkl")
        return model
    except Exception as e:
        st.error(f" Error loading model: {e}")
        return None

model = load_model()

st.title("Water Potability Prediction App")
st.write("Check whether water is safe to drink using Machine Learning")


st.sidebar.header("Input Water Parameters")

def user_input():
    ph = st.sidebar.slider("pH value", 0.0, 14.0, 7.0)
    hardness = st.sidebar.slider("Hardness", 0.0, 500.0, 200.0)
    solids = st.sidebar.slider("Total Dissolved Solids", 0.0, 50000.0, 15000.0)
    chloramines = st.sidebar.slider("Chloramines", 0.0, 15.0, 7.0)
    sulfate = st.sidebar.slider("Sulfate", 0.0, 500.0, 250.0)
    conductivity = st.sidebar.slider("Conductivity", 0.0, 1000.0, 400.0)
    organic_carbon = st.sidebar.slider("Organic Carbon", 0.0, 30.0, 10.0)
    trihalomethanes = st.sidebar.slider("Trihalomethanes", 0.0, 150.0, 70.0)
    turbidity = st.sidebar.slider("Turbidity", 0.0, 10.0, 4.0)

    data = pd.DataFrame([{
        "ph": ph,
        "Hardness": hardness,
        "Solids": solids,
        "Chloramines": chloramines,
        "Sulfate": sulfate,
        "Conductivity": conductivity,
        "Organic_carbon": organic_carbon,
        "Trihalomethanes": trihalomethanes,
        "Turbidity": turbidity
    }])

    return data

input_data = user_input()

st.subheader("Input Parameters")
st.dataframe(input_data)

if st.button(" Predict Potability"):
    if model is not None:
        try:
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)

            st.subheader("Result")

            if prediction[0] == 1:
                st.success("Water is POTABLE (Safe to Drink)")
            else:
                st.error("Water is NOT Potable (Unsafe)")

            st.subheader("Prediction Probability")
            st.write(f"Not Potable: {probability[0][0]:.2f}")
            st.write(f"Potable: {probability[0][1]:.2f}")

        except Exception as e:
            st.error(f"Prediction Error: {e}")
    else:
        st.warning("Model not loaded properly.")