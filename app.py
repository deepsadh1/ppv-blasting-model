import streamlit as st
import numpy as np
import pandas as pd
import joblib
from io import BytesIO

# Load model
model = joblib.load("final_xgboost_model.pkl")

st.set_page_config(page_title="PPV Prediction | IIT BHU", layout="centered")

# Title
st.markdown("<h1 style='text-align: center; color: black;'>💥 Peak Particle Velocity Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: black;'>XGBoost-Based Blasting Evaluation Model</h3>", unsafe_allow_html=True)

# --- Single Prediction ---
st.markdown("### 🔢 Predict for Single Blast Event")
distance = st.number_input("📝 Distance from Blast (m)", min_value=1.0, value=100.0)
charge = st.number_input("💣 Charge per Delay (kg)", min_value=1.0, value=50.0)
rock_type = st.selectbox("🪨 Select Rock Type", ["Coal", "Limestone"])

if st.button("🧠 Predict PPV"):
    scaled_distance = distance / np.sqrt(charge)
    distance_x_charge = distance * charge
    distance_squared = distance ** 2
    charge_squared = charge ** 2

    # Create one-hot encoding
    rock_coal = 1 if rock_type.lower() == "coal" else 0
    rock_limestone = 1 if rock_type.lower() == "limestone" else 0

    features = pd.DataFrame([{
        "Distance (m)": distance,
        "Charge per Delay (kg)": charge,
        "Charge_squared": charge_squared,
        "Distance_squared": distance_squared,
        "Distance_x_Charge": distance_x_charge,
        "Rock_coal": rock_coal,
        "Rock_limestone": rock_limestone
    }])

    predicted_ppv = model.predict(features)[0]
    ppv_value = round(predicted_ppv, 3)

    st.success(f"📈 Predicted PPV: {ppv_value} mm/s")

    if ppv_value < 5:
        st.markdown('<div style="color:green;font-weight:bold;">🟢 Safe Zone (PPV < 5 mm/s)</div>', unsafe_allow_html=True)
    elif 5 <= ppv_value <= 10:
        st.markdown('<div style="color:orange;font-weight:bold;">🟠 Moderate Risk (5 ≤ PPV ≤ 10 mm/s)</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:red;font-weight:bold;">🔴 Danger Zone (PPV > 10 mm/s)</div>', unsafe_allow_html=True)

# --- Batch Upload ---
st.markdown("---")
st.markdown("### 📂 Upload CSV for Batch Prediction")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    try:
        df_raw = pd.read_csv(uploaded_file)
        df = df_raw.copy()
        df.columns = df.columns.str.strip()

        # Match columns
        distance_col = next((col for col in df.columns if 'distance' in col.lower()), None)
        charge_col = next((col for col in df.columns if 'charge' in col.lower()), None)
        rock_col = next((col for col in df.columns if 'rock' in col.lower()), None)

        if not all([distance_col, charge_col, rock_col]):
            st.error("CSV must contain columns for Distance, Charge, and Rock Type.")
        else:
            df = df[[distance_col, charge_col, rock_col]].copy()
            df.columns = ['Distance (m)', 'Charge per Delay (kg)', 'Rock Type']

            df["Charge_squared"] = df["Charge per Delay (kg)"] ** 2
            df["Distance_squared"] = df["Distance (m)"] ** 2
            df["Distance_x_Charge"] = df["Distance (m)"] * df["Charge per Delay (kg)"]

            df["Rock_coal"] = df["Rock Type"].str.lower().apply(lambda x: 1 if x == "coal" else 0)
            df["Rock_limestone"] = df["Rock Type"].str.lower().apply(lambda x: 1 if x == "limestone" else 0)

            input_features = df[[
                "Distance (m)", "Charge per Delay (kg)",
                "Charge_squared", "Distance_squared",
                "Distance_x_Charge", "Rock_coal", "Rock_limestone"
            ]]

            df_raw['Predicted PPV (mm/s)'] = model.predict(input_features).round(3)

            st.markdown("#### ✅ Prediction Preview")
            st.dataframe(df_raw.head())

            output = BytesIO()
            df_raw.to_csv(output, index=False)
            st.download_button("📥 Download Predicted CSV", output.getvalue(), "predicted_ppv_output.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

# --- Footer ---
st.markdown("""
<hr style="margin-top:50px;">
<div style="text-align:center;color:gray;font-size:0.9em">
    Built with ❤️ by <b>IIT BHU</b> Mining Engineering <br>
    Powered by Streamlit + XGBoost
</div>
""", unsafe_allow_html=True)
