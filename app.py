import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO

# Load the trained XGBoost model
model = joblib.load("final_xgboost_model.pkl")

# Streamlit page setup
st.set_page_config(page_title="PPV Prediction | IIT BHU", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    .indicator {
        font-size: 1.2rem;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        width: 100%;
    }
    .safe { background-color: #d4edda; color: #155724; }
    .moderate { background-color: #fff3cd; color: #856404; }
    .danger { background-color: #f8d7da; color: #721c24; }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center;'>💥 Peak Particle Velocity Prediction</h1>", unsafe_allow_html=True)
st.subheader("XGBoost-Based Blasting Evaluation Model")

# ----------- Manual Input Section -----------
st.markdown("### 🎯 Predict for a Single Input")

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

    if ppv_value < 5:
        safety_html = '<div class="indicator safe">🟢 Safe Zone (PPV < 5 mm/s)</div>'
    elif 5 <= ppv_value <= 10:
        safety_html = '<div class="indicator moderate">🟠 Moderate Risk (5 ≤ PPV ≤ 10 mm/s)</div>'
    else:
        safety_html = '<div class="indicator danger">🔴 Danger Zone (PPV > 10 mm/s)</div>'
    st.markdown(safety_html, unsafe_allow_html=True)

# ----------- Batch CSV Prediction -----------
st.markdown("### 📂 Upload CSV for Batch Prediction")
uploaded_file = st.file_uploader("Upload CSV file (must include 'Distance', 'Charge', and 'Rock Type')", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Normalize column names
        df.columns = df.columns.str.strip().str.lower()

        required_cols = ['distance', 'charge', 'rock type']
        if all(col in df.columns for col in required_cols):

            # Drop extra columns
            df_trimmed = df[required_cols].copy()

            # Encode rock type
            df_trimmed['rock'] = df_trimmed['rock type'].str.lower().map({'coal': 0, 'limestone': 1})

            # Feature engineering
            df_trimmed['scaled_distance'] = df_trimmed['distance'] / np.sqrt(df_trimmed['charge'])
            df_trimmed['distance_x_charge'] = df_trimmed['distance'] * df_trimmed['charge']
            df_trimmed['distance_squared'] = df_trimmed['distance'] ** 2
            df_trimmed['charge_squared'] = df_trimmed['charge'] ** 2

            X = df_trimmed[['distance', 'charge', 'scaled_distance', 'rock', 'distance_x_charge', 'distance_squared', 'charge_squared']]
            df['Predicted PPV (mm/s)'] = model.predict(X).round(3)

            st.markdown("#### ✅ Preview with Predictions")
            st.dataframe(df.head())

            # Download button
            output = BytesIO()
            df.to_csv(output, index=False)
            st.download_button(
                label="📥 Download Predicted CSV",
                data=output.getvalue(),
                file_name="predicted_ppv_output.csv",
                mime="text/csv"
            )
        else:
            st.error("CSV must include 'Distance', 'Charge', and 'Rock Type' columns.")

    except Exception as e:
        st.error(f"Error processing file: {e}")

# Footer
st.markdown("""
<hr style="margin-top:50px;">
<div style="text-align:center;color:gray;font-size:0.9em">
    Built with ❤️ by <b>IIT BHU</b> Mining Engineering <br>
    Powered by Streamlit + XGBoost
</div>
""", unsafe_allow_html=True)
