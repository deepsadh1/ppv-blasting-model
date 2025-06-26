import streamlit as st
import numpy as np
import pandas as pd
import joblib
from io import BytesIO

# Load model
model = joblib.load("final_xgboost_model.pkl")

# Page config
st.set_page_config(page_title="PPV Prediction | IIT BHU", layout="centered")

# Title and subtitle
st.markdown("<h1 style='text-align: center; color: black;'>💥 Peak Particle Velocity Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: black;'>XGBoost-Based Blasting Evaluation Model</h3>", unsafe_allow_html=True)

# --- Individual Prediction Form ---
st.markdown("### 🔢 Predict for Single Blast Event")

distance = st.number_input("📝 Distance from Blast (m)", min_value=1.0, value=100.0)
charge = st.number_input("💣 Charge per Delay (kg)", min_value=1.0, value=50.0)
rock_type = st.selectbox("🪨 Select Rock Type", ["Coal", "Limestone"])
rock = 0 if rock_type == "Coal" else 1

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

    # Safety label
    if ppv_value < 5:
        st.markdown('<div style="color:green;font-weight:bold;">🟢 Safe Zone (PPV < 5 mm/s)</div>', unsafe_allow_html=True)
    elif 5 <= ppv_value <= 10:
        st.markdown('<div style="color:orange;font-weight:bold;">🟠 Moderate Risk (5 ≤ PPV ≤ 10 mm/s)</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:red;font-weight:bold;">🔴 Danger Zone (PPV > 10 mm/s)</div>', unsafe_allow_html=True)

# --- Batch CSV Upload Section ---
st.markdown("---")
st.markdown("### 📂 Upload CSV for Batch Prediction")
uploaded_file = st.file_uploader("Upload CSV file (must include Distance, Charge, and Rock Type columns)", type=["csv"])

if uploaded_file:
    try:
        df_raw = pd.read_csv(uploaded_file)
        df = df_raw.copy()
        df.columns = df.columns.str.strip().str.lower()

        # Match columns flexibly
        col_map = {}
        for col in df.columns:
            if 'distance' in col:
                col_map['distance'] = col
            elif 'charge' in col:
                col_map['charge'] = col
            elif 'rock' in col and 'type' in col:
                col_map['rock type'] = col

        # Validate presence
        if set(col_map.keys()) >= {'distance', 'charge', 'rock type'}:
            df_trim = df[[col_map['distance'], col_map['charge'], col_map['rock type']]].copy()
            df_trim.columns = ['distance', 'charge', 'rock type']

            # Encode rock type
            df_trim['rock'] = df_trim['rock type'].str.lower().map({'coal': 0, 'limestone': 1})

            # Feature engineering
            df_trim['scaled_distance'] = df_trim['distance'] / np.sqrt(df_trim['charge'])
            df_trim['distance_x_charge'] = df_trim['distance'] * df_trim['charge']
            df_trim['distance_squared'] = df_trim['distance'] ** 2
            df_trim['charge_squared'] = df_trim['charge'] ** 2

            X = df_trim[['distance', 'charge', 'scaled_distance', 'rock',
                         'distance_x_charge', 'distance_squared', 'charge_squared']]

            df_raw['Predicted PPV (mm/s)'] = model.predict(X).round(3)

            st.markdown("#### ✅ Prediction Preview")
            st.dataframe(df_raw.head())

            # Download button
            output = BytesIO()
            df_raw.to_csv(output, index=False)
            st.download_button(
                label="📥 Download Predicted CSV",
                data=output.getvalue(),
                file_name="predicted_ppv_output.csv",
                mime="text/csv"
            )
        else:
            st.error("CSV must include columns for Distance, Charge, and Rock Type.")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

# Footer
st.markdown("""
<hr style="margin-top:50px;">
<div style="text-align:center;color:gray;font-size:0.9em">
    Built with ❤️ by <b>IIT BHU</b> Mining Engineering <br>
    Powered by Streamlit + XGBoost
</div>
""", unsafe_allow_html=True)
