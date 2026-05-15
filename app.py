"""
Used Vehicle Price Predictor — Streamlit UI
Capstone Project: Tinka Fahad & Tugume Andrew
Cavendish University Uganda, May 2026
"""

import streamlit as st
import joblib
import json
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Used Vehicle Price Predictor — Capstone",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def load_model_and_metadata():
    pipeline = joblib.load('rf_pipeline.joblib')
    with open('ui_metadata.json', 'r') as f:
        metadata = json.load(f)
    return pipeline, metadata

@st.cache_data
def load_cleaned_data():
    return pd.read_csv('vehicles_clean.csv')

try:
    pipeline, meta = load_model_and_metadata()
    df_clean = load_cleaned_data()
except FileNotFoundError as e:
    st.error(f"Required file not found: {e.filename}. Run the notebook first.")
    st.stop()

INR_TO_UGX = 45
INR_TO_USD = 0.012

with st.sidebar:
    st.markdown("## 📚 About this project")
    st.markdown(
        """
        **Capstone Project**
        Predictive Analytics

        **Authors:**
        - Tinka Fahad (254430)
        - Tugume Andrew

        **Institution:**
        Cavendish University Uganda
        BSc Data Science & AI

        **Date:** May 2026
        """
    )
    st.markdown("---")
    st.markdown("## 🎯 Model performance")
    metrics = meta.get('model_metrics', {})
    if metrics:
        st.metric("R² Score", f"{metrics.get('r2', 0):.3f}")
        st.metric("RMSE", f"₹{metrics.get('rmse_inr', 0):,.0f}")
        st.metric("MAE", f"₹{metrics.get('mae_inr', 0):,.0f}")
    st.markdown("---")
    st.markdown("## 📊 Dataset")
    st.markdown(
        f"""
        - **Source:** CarDekho India (Kaggle)
        - **Records:** {len(df_clean):,}
        - **Model:** Random Forest Regression
        """
    )

st.title("🚗 Used Vehicle Price Predictor")
st.markdown(
    """
    A Random Forest regression model predicting used vehicle prices,
    trained on the **CarDekho India** dataset. Designed for application to
    Uganda's used-vehicle market, where similar manufacturers
    (Maruti-Suzuki, Toyota, Honda, Hyundai) dominate the ride-hailing fleet.
    """
)
st.markdown("---")
st.markdown("## 🔧 Enter vehicle details")

col1, col2, col3 = st.columns(3)

with col1:
    manufacturer = st.selectbox(
        "Manufacturer", meta['manufacturers'],
        index=meta['manufacturers'].index('Toyota') if 'Toyota' in meta['manufacturers'] else 0
    )
    year = st.number_input(
        "Year of manufacture", min_value=meta['year_min'], max_value=meta['year_max'],
        value=min(2014, meta['year_max']), step=1
    )
    km_driven = st.number_input(
        "Kilometres driven", min_value=0, max_value=meta['km_driven_max'],
        value=80_000, step=1000
    )

with col2:
    fuel = st.selectbox("Fuel type", meta['fuel_types'])
    transmission = st.selectbox("Transmission", meta['transmissions'])
    owner = st.selectbox("Ownership history", meta['owner_types'])

with col3:
    engine_cc = st.number_input(
        "Engine size (CC)", min_value=meta['engine_cc_min'], max_value=meta['engine_cc_max'],
        value=1500, step=100
    )
    max_power = st.number_input(
        "Max power (bhp)", min_value=float(meta['max_power_min']), max_value=float(meta['max_power_max']),
        value=85.0, step=5.0
    )
    mileage = st.number_input(
        "Fuel efficiency (kmpl)", min_value=float(meta['mileage_min']), max_value=float(meta['mileage_max']),
        value=18.0, step=0.5
    )

col4, col5 = st.columns(2)
with col4:
    seats = st.selectbox(
        "Number of seats", meta['seats_options'],
        index=meta['seats_options'].index(5) if 5 in meta['seats_options'] else 0
    )
with col5:
    seller_type = st.selectbox("Seller type", meta['seller_types'])

st.markdown("---")

if st.button("📈 Predict price", type="primary", use_container_width=True):
    age = 2021 - year
    input_df = pd.DataFrame([{
        'age': age, 'km_driven': km_driven, 'mileage_kmpl': mileage,
        'engine_cc': engine_cc, 'max_power_bhp': max_power, 'seats': seats,
        'manufacturer': manufacturer, 'fuel': fuel, 'transmission': transmission,
        'seller_type': seller_type, 'owner': owner,
    }])

    predicted_price_inr = float(pipeline.predict(input_df)[0])
    predicted_price_ugx = predicted_price_inr * INR_TO_UGX
    predicted_price_usd = predicted_price_inr * INR_TO_USD

    st.markdown("## 💰 Predicted price")
    st.metric("Ugandan Shillings", f"UGX {predicted_price_ugx:,.0f}")
    st.markdown(
        f"<p style='text-align: center; color: #666; font-size: 0.9em;'>"
        f"Reference: ₹{predicted_price_inr:,.0f} INR · ${predicted_price_usd:,.0f} USD"
        f"</p>",
        unsafe_allow_html=True
    )
    st.caption(
        "_Currency conversions are approximate (May 2026 rates) for cross-market intuition. "
        "Actual Ugandan prices depend on import duties, local supply, and listing-specific "
        "factors not captured in this model._"
    )

    st.markdown("### 🔍 How this compares to similar vehicles in the dataset")
    similar = df_clean[
        (df_clean['manufacturer'] == manufacturer) &
        (df_clean['fuel'] == fuel) &
        (df_clean['age'].between(max(0, age - 2), age + 2))
    ]

    if len(similar) >= 5:
        avg_price = similar['selling_price'].mean()
        median_price = similar['selling_price'].median()
        avg_price_ugx = avg_price * INR_TO_UGX
        median_price_ugx = median_price * INR_TO_UGX
        pct_diff = ((predicted_price_inr - avg_price) / avg_price) * 100

        st.markdown(f"Found **{len(similar)} similar vehicles** ({manufacturer}, {fuel}, age ±2 years).")
        c1, c2 = st.columns(2)
        c1.metric("Average of similar", f"UGX {avg_price_ugx:,.0f}", help=f"₹ {avg_price:,.0f} INR")
        c2.metric("Median of similar", f"UGX {median_price_ugx:,.0f}", help=f"₹ {median_price:,.0f} INR")

        if abs(pct_diff) < 10:
            st.success(f"✅ Predicted price within 10% of average for similar vehicles ({pct_diff:+.1f}%) — typical for this profile.")
        elif pct_diff > 0:
            st.info(f"📈 Predicted price {pct_diff:+.1f}% above average. May reflect higher power, lower km, or premium ownership history.")
        else:
            st.warning(f"📉 Predicted price {pct_diff:+.1f}% below average. May reflect high km, older ownership, or smaller engine.")
    else:
        st.info("Not enough similar vehicles for comparison. Prediction is based on broader model patterns.")

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888; font-size: 0.9em;'>
        Predictive Analytics Capstone Project. Random Forest Regression trained on CarDekho India dataset.
        Methodology designed for transfer to Uganda's used-vehicle market.
    </div>
    """,
    unsafe_allow_html=True
)
