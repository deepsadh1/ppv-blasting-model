import streamlit as st
import numpy as np
import joblib

# Load the model
model = joblib.load("final_xgboost_model.pkl")

# Page config
st.set_page_config(page_title="PPV Prediction | IIT BHU", layout="centered")

# Custom styles: background + logo + fonts
st.markdown("""
    <style>
    .stApp {
        background-image: url('https://raw.githubusercontent.com/deepsadh1/ppv-blasting-model/main/background_image.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: white;
    }
    h1, h2, h3, h4, h5, h6, .stTextInput label, .stSelectbox label, .stNumberInput label {
        color: white !important;
    }
    .logo-container {
        position: absolute;
        top: 15px;
        right: 25px;
        z-index: 100;
    }
    .logo-container img {
        width: 100px;
        height: auto;
    }
    .indicator {
        font-size: 1.2rem;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        width: 100%;
    }
    .safe {
        background-color: #d4edda;
        color: #155724;
    }
    .moderate {
        background-color: #fff3cd;
        color: #856404;
    }
    .danger {
        background-color: #f8d7da;
        color: #721c24;
    }
    </style>
""", unsafe_allow_html=True)

# IIT BHU Logo
st.markdown("""
    <div class="logo-container">
        <img src="https://raw.githubusercontent.com/deepsadh1/ppv-blasting-model/main/iit_bhu_logo.png">
    </div>
""", unsafe_allow_html=True)

# Title and Subtitle
st.markdown("<h1 style='text-align: center;'>💥 Peak Particle Velocity Prediction</h1>", unsafe_allow_html=True)
st.subheader("XGBoost-Based Blasting Evaluation Model")

# Inputs
distance = st.number_input("📝 Distance from Blast (m)", min_value=1.0, value=100.0)
charge = st.number_input("💣 Charge per Delay (kg)", min_value=1.0, value=50.0)
rock_type = st.selectbox("🪨 Select Rock Type", ["Coal", "Limestone"])

# Rock encoding
rock = 0 if rock_type == "Coal" else 1

# Prediction
if st.button("🧠 Predict PPV"):
    scaled_distance = distance / np.sqrt(charge)
    distance_x_charge = distance * charge
    distance_squared = distance ** 2
    charge_squared = charge ** 2

    features = np.array([[
        distance, charge, scaled_distance, rock,
        distance_x_charge, distance_squared, charge_squared
    ]])

    predicted_ppv = model.predict(features)[0]
    ppv_value = round(predicted_ppv, 3)
    
    st.success(f"📈 Predicted PPV: {ppv_value} mm/s")

    # Safety Indicator
    if ppv_value < 5:
        safety_html = '<div class="indicator safe">🟢 Safe Zone (PPV < 5 mm/s)</div>'
    elif 5 <= ppv_value <= 10:
        safety_html = '<div class="indicator moderate">🟠 Moderate Risk (5 ≤ PPV ≤ 10 mm/s)</div>'
    else:
        safety_html = '<div class="indicator danger">🔴 Danger Zone (PPV > 10 mm/s)</div>'

    st.markdown(safety_html, unsafe_allow_html=True)

# Footer
st.markdown("""
<hr style="margin-top:50px;">
<div style="text-align:center;color:gray;font-size:0.9em">
    Built with ❤️ by <b>IIT BHU</b> Mining Engineering <br>
    Powered by Streamlit + XGBoost
</div>
""", unsafe_allow_html=True)
