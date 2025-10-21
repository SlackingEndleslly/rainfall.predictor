import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model & dataset
pipeline = joblib.load("rainfall_pipeline.pkl")
df = pd.read_csv("district-wise-rainfall-normal.csv")

st.set_page_config(page_title="Rainfall Predictor", page_icon="☔", layout="wide")

# Custom background and style
st.markdown("""
    <style>
    /* Main background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #1C1F26 0%, #2C3E50 100%);
        font color: #E8E8E8;
        accent: #56CCF2 (soft aqua for buttons and highlights)
        font-family: 'Poppins', sans-serif;
        padding-top: 20px;
    }

    /* Title style */
    h1 {
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        color: #ffffff !important;
        margin-bottom: 10px;
    }

    /* Input sections */
    div[data-testid="stSlider"] > div {
        background-color: rgba(255, 255, 255, 0.15);
        border-radius: 10px;
        padding: 5px;
    }

    /* Prediction Button */
    div.stButton > button {
        background-color: #ffffff;
        color: #333333;
        border-radius: 12px;
        font-size: 18px;
        font-weight: 600;
        height: 3em;
        width: 100%;
        transition: 0.3s;
    }

    div.stButton > button:hover {
        background-color: #ffd700;
        color: #1a1a1a;
    }

    /* Result box */
    .stSuccess {
        background-color: rgba(255,255,255,0.2);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: #fff !important;
    }

    /* Metric styling */
    div[data-testid="stMetricValue"] {
        color: #ffd700;
        font-weight: 700;
        font-size: 2rem;
    }

    footer, header {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# App Header
st.title("🌦️ Seasonal Rainfall Predictor")

# State and District Input
col1, col2 = st.columns(2)
with col1:
    states = df['STATE_UT_NAME'].unique()
    selected_state = st.selectbox('', options=sorted(states), index=0, label_visibility="hidden", key="state")
    st.subheader("State / UT")

with col2:
    districts = df[df['STATE_UT_NAME'] == selected_state]['DISTRICT'].unique()
    selected_district = st.selectbox('', options=sorted(districts), index=0, label_visibility="hidden", key="district")
    st.subheader("District")

# Monthly Rainfall Inputs
st.subheader("Monthly Rainfall (mm)")
months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
cols = st.columns(4)
input_data = {}
for i, month in enumerate(months):
    with cols[i % 4]:
        input_data[month] = st.slider(f"{month}", min_value=0, max_value=1000, value=100)

# Feature engineering
input_data['Winter_Avg'] = np.mean([input_data['DEC'], input_data['JAN'], input_data['FEB']])
input_df = pd.DataFrame([{**input_data, 'STATE_UT_NAME': selected_state, 'DISTRICT': selected_district}])

# Predict Button
st.write("")  # spacer
if st.button("💧 Predict Monsoon Rainfall"):
    try:
        pred = pipeline.predict(input_df)[0]
        st.success(f"☔ Predicted Jun–Sep Rainfall: {pred:.2f} mm")
        st.metric(label="Expected Monsoon Rainfall", value=f"{pred:.2f} mm")
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
