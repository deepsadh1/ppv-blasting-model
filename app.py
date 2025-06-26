import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the model
model = joblib.load("final_xgboost_model.pkl")

# Page config
st.set_page_config(page_title="PPV Prediction | IIT BHU", layout="centered")

# Title
st.markdown("<h1 style='text-align: center;'>💥 Peak Particle Velocity Prediction</h1>", unsafe_allow_html=True)
st.subheader("XGBoost-Based Blasting Evaluation Model")

# --- SINGLE INPUT PREDICTION ---
st.markdown("### 🔍 Predict from Single Input")

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

    st.markdown(f"""
        <style>
        .indicator {{
            font-size: 1.2rem;
            font-weight: bold;
            padding: 0.5rem;
            border-radius: 5px;
            text-align: center;
        }}
        .safe {{ background-color: #d4edda; color: #155724; }}
        .moderate {{ background-color: #fff3cd; color: #856404; }}
        .danger {{ background-color: #f8d7da; color: #721c24; }}
        </style>
        {safety_html}
    """, unsafe_allow_html=True)

# --- CSV UPLOAD FOR BATCH PREDICTION ---
st.markdown("---")
st.subheader("📂 Upload CSV for Batch Prediction")

uploaded_file = st.file_uploader("Upload a CSV file (columns: Distance, Charge, Rock Type)", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.write("📄 Preview of Uploaded Data:")
        st.dataframe(df.head())

        if all(col in df.columns for col in ['Distance', 'Charge', 'Rock Type']):
            df['Rock Type Encoded'] = df['Rock Type'].apply(lambda x: 0 if x.lower() == 'coal' else 1)
            df['Scaled Distance'] = df['Distance'] / np.sqrt(df['Charge'])
            df['Distance x Charge'] = df['Distance'] * df['Charge']
            df['Distance^2'] = df['Distance'] ** 2
            df['Charge^2'] = df['Charge'] ** 2

            feature_cols = ['Distance', 'Charge', 'Scaled Distance', 'Rock Type Encoded',
                            'Distance x Charge', 'Distance^2', 'Charge^2']
            df['Predicted PPV'] = model.predict(df[feature_cols])
            df['Predicted PPV'] = df['Predicted PPV'].round(3)

            def classify(ppv):
                if ppv < 5:
                    return "Safe"
                elif 5 <= ppv <= 10:
                    return "Moderate"
                else:
                    return "Dangerous"

            df['Safety Status'] = df['Predicted PPV'].apply(classify)

            st.success("✅ Predictions Generated")
            st.dataframe(df[['Distance', 'Charge', 'Rock Type', 'Predicted PPV', 'Safety Status']])

            csv_output = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions", csv_output, "ppv_predictions.csv", "text/csv")
        else:
            st.error("❗ CSV must include 'Distance', 'Charge', and 'Rock Type' columns.")
    except Exception as e:
        st.error(f"❌ Error reading CSV: {e}")

# --- Footer ---
st.markdown("""
<hr style="margin-top:50px;">
<div style="text-align:center;color:gray;font-size:0.9em">
    Built with ❤️ by <b>IIT BHU</b> Mining Engineering <br>
    Powered by Streamlit + XGBoost
</div>
""", unsafe_allow_html=True)
