
import streamlit as st
import numpy as np
import joblib

# Load the model
model = joblib.load("final_xgboost_model.pkl")

# Page config
st.set_page_config(page_title="PPV Prediction | IIT BHU", layout="wide")

# Inject CSS for local background and IIT BHU logo
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
    <style>
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    .stApp {
        background-image: url('background_image.jpg');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .logo-container {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        padding-right: 20px;
        padding-top: 10px;
    }
    .logo-container img {
        width: 100px;
    }
    .indicator {
        font-size: 1.2rem;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 8px;
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

# Display logo
st.markdown("""
    <div class="logo-container">
        <img src="iit_bhu_logo.png" alt="IIT BHU Logo">
    </div>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown("<h1 style='text-align: center;'>💥 Peak Particle Velocity Prediction</h1>", unsafe_allow_html=True)
st.subheader("XGBoost-Based Blasting Evaluation Model")

# Sidebar input panel
with st.sidebar:
    st.header("🔧 Input Parameters")
    distance = st.number_input("📏 Distance from Blast (m)", min_value=1.0, value=100.0, help="Distance from blast point to sensor")
    charge = st.number_input("💣 Charge per Delay (kg)", min_value=1.0, value=50.0, help="Explosive charge per delay")
    rock_type = st.selectbox("🪨 Select Rock Type", ["Coal", "Limestone"], help="Type of rock formation")

# Rock encoding
rock = 0 if rock_type == "Coal" else 1

# Prediction area
st.markdown("## 🔍 Prediction Result")

if st.button("🧠 Predict PPV"):
    with st.spinner("Calculating PPV..."):
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

        if ppv_value < 5:
            safety_html = '<div class="indicator safe">🟢 Safe Zone (PPV < 5 mm/s)</div>'
        elif 5 <= ppv_value <= 10:
            safety_html = '<div class="indicator moderate">🟠 Moderate Risk (5 ≤ PPV ≤ 10 mm/s)</div>'
        else:
            safety_html = '<div class="indicator danger">🔴 Danger Zone (PPV > 10 mm/s)</div>'

        st.markdown(f"""
            <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;margin-top:20px">
                <h3 style="color:#333;">📈 Predicted PPV: <span style="color:#007BFF;">{ppv_value} mm/s</span></h3>
                {safety_html}
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<hr style="margin-top:50px;">
<div style="text-align:center;color:gray;font-size:0.9em">
    Built with ❤️ by <b>IIT BHU</b> Mining Engineering <br>
    Powered by Streamlit + XGBoost
</div>
""", unsafe_allow_html=True)
