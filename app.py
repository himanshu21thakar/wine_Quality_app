import streamlit as st
import joblib
import numpy as np

# --- Page Config ---
st.set_page_config(page_title="🍷 Wine Quality Classifier", page_icon="🍷", layout="centered")

# --- Dynamic CSS ---
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #f8f9fa, #e3f2fd);
        font-family: Arial, sans-serif;
    }
    .main {
        padding: 2rem;
        background-color: #ffffffcc;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    }
    h1, h2 {
        text-align: center;
        color: #2c3e50;
    }
    .stButton>button {
        background-color: #4cafef;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
        cursor: pointer;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #2196f3;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Load Model and Scaler ---
try:
    model = joblib.load("wine_svm_model.pkl")
    scaler = joblib.load("wine_scaler.pkl")
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# --- App Title ---
st.title("🍷 Wine Quality Classifier")
st.write("Enter the wine characteristics below to predict if it's **Good** (≥6) or **Not Good** (<6).")

# --- Feature Inputs ---
col1, col2 = st.columns(2)

fixed_acidity = col1.number_input("Fixed Acidity", min_value=0.0, max_value=20.0, value=7.4)
volatile_acidity = col2.number_input("Volatile Acidity", min_value=0.0, max_value=2.0, value=0.7)
citric_acid = col1.number_input("Citric Acid", min_value=0.0, max_value=1.0, value=0.0)
residual_sugar = col2.number_input("Residual Sugar", min_value=0.0, max_value=15.0, value=1.9)
chlorides = col1.number_input("Chlorides", min_value=0.0, max_value=1.0, value=0.076)
free_sulfur_dioxide = col2.number_input("Free Sulfur Dioxide", min_value=0.0, max_value=80.0, value=11.0)
total_sulfur_dioxide = col1.number_input("Total Sulfur Dioxide", min_value=0.0, max_value=300.0, value=34.0)
density = col2.number_input("Density", min_value=0.990, max_value=1.005, value=0.9978)
ph = col1.number_input("pH", min_value=2.5, max_value=4.5, value=3.51)
sulphates = col2.number_input("Sulphates", min_value=0.3, max_value=2.0, value=0.56)
alcohol = st.slider("Alcohol", min_value=8.0, max_value=15.0, value=9.4)

# --- Prediction Button ---
if st.button("Predict Quality"):
    features = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                          free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    label = "🍇 **Good Quality Wine!**" if prediction[0] == 1 else "⚠️ **Not Good Quality Wine.**"

    st.subheader("Prediction Result")
    st.success(label)

# --- Footer ---
st.markdown("<hr style='border:1px solid #ddd'>", unsafe_allow_html=True)
st.caption("Made with ❤️ using Streamlit and SVM")
