"""
Used Vehicle Price Predictor — Streamlit UI
Capstone Project: Tinka Fahad & Tugume Andrew
Cavendish University Uganda, May 2026
"""

import streamlit as st
import joblib
import json
import pandas as pd

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

tabs = st.tabs(["🔮 Predict Price", "📊 Explore the Data", "📈 Model Comparison"])

with tabs[0]:
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
        st.markdown(
            "*Prediction based on Random Forest model trained on 7,857 vehicles. "
            "Model achieves R² = 0.969 on held-out test data.*"
        )
        st.caption(
            "_Currency conversions are approximate (May 2026 rates) for cross-market intuition. "
            "Actual Ugandan prices depend on import duties, local supply, and listing-specific "
            "factors not captured in this model._"
        )

        st.markdown("### 🔍 How this compares to similar vehicles in the dataset")
        similar = df_clean[
            (df_clean['manufacturer'] == manufacturer) &
            (df_clean['year'].between(max(0, year - 2), year + 2)) &
            (df_clean['km_driven'].between(max(0, km_driven - 30000), km_driven + 30000))
        ]

        if len(similar) < 5:
            similar = df_clean[
                (df_clean['manufacturer'] == manufacturer) &
                (df_clean['year'].between(max(0, year - 4), year + 4))
            ]

        if len(similar) < 5:
            similar = df_clean[df_clean['manufacturer'] == manufacturer]

        if len(similar) < 3:
            similar = df_clean[
                (df_clean['fuel'] == fuel) &
                (df_clean['year'].between(max(0, year - 3), year + 3))
            ]

        if len(similar) < 3:
            st.info("Not enough similar vehicles for comparison. Prediction is based on broader model patterns.")
        else:
            avg_price = similar['selling_price'].mean()
            median_price = similar['selling_price'].median()
            avg_price_ugx = avg_price * INR_TO_UGX
            median_price_ugx = median_price * INR_TO_UGX
            diff_pct = ((predicted_price_inr - median_price) / median_price) * 100

            st.markdown(f"Compared against **{len(similar)} similar vehicles** in the dataset")
            c1, c2 = st.columns(2)
            c1.metric("Average of similar", f"UGX {avg_price_ugx:,.0f}", help=f"₹ {avg_price:,.0f} INR")
            c2.metric("Median of similar", f"UGX {median_price_ugx:,.0f}", help=f"₹ {median_price:,.0f} INR")

            if abs(diff_pct) <= 15:
                st.success("✅ Your prediction is in line with similar vehicles.")
            elif diff_pct > 15:
                st.info("📈 Your prediction is higher than typical.")
            else:
                st.warning("📉 Your prediction is lower than typical.")

with tabs[1]:
    st.markdown("## 📊 Explore the Data")
    st.markdown(
        "*This section explores the full cleaned CarDekho dataset and is independent of the prediction inputs. "
        "Use it to understand the data the model was trained on.*"
    )
    st.markdown(
        "This is the cleaned CarDekho dataset (~7,857 records) used for model training and analysis. "
        "Use the charts and descriptive statistics to understand the core structure of the dataset."
    )

    stats_cols = [
        'selling_price', 'km_driven', 'age', 'mileage_kmpl',
        'engine_cc', 'max_power_bhp', 'seats'
    ]
    stats = pd.DataFrame({
        'mean': df_clean[stats_cols].mean(),
        'median': df_clean[stats_cols].median(),
        'mode': df_clean[stats_cols].mode().iloc[0],
        'std': df_clean[stats_cols].std(),
        'count': df_clean[stats_cols].count(),
    })
    stats = stats[['mean', 'median', 'mode', 'std', 'count']]
    stats.index.name = 'feature'
    st.dataframe(stats.style.format({
        'mean': '{:,.2f}',
        'median': '{:,.2f}',
        'mode': '{:,.2f}',
        'std': '{:,.2f}',
        'count': '{:,.0f}'
    }))

    st.image('fig_fuel_pie.png', caption='Distribution of vehicle fuel types', width='stretch')
    st.image('fig_top_manufacturers.png', caption="Top 15 manufacturers — same brands dominate Uganda's used-car fleet", width='stretch')
    st.image('fig_age_km_distributions.png', caption='Age and kilometres driven distributions', width='stretch')
    st.image('fig_price_distribution.png', caption='Price distribution — right-skewed, motivating log-transform', width='stretch')
    st.image('fig_correlation_heatmap.png', caption='Correlation between numeric features', width='stretch')
    st.image('fig_scatter_relationships.png', caption='Price vs age and kilometres — clearly non-linear, motivating polynomial regression', width='stretch')

with tabs[2]:
    st.markdown("## 📈 Model Comparison")
    st.markdown(
        "*This section evaluates all three regression models on the held-out test set. Metrics are computed once during training "
        "and do not change with prediction inputs.*"
    )
    st.markdown("Comparison of three regression models trained on the cleaned dataset")

    comparison_df = pd.read_csv('model_comparison.csv')
    st.table(comparison_df)

    st.image('fig_model_comparison.png', caption='Side-by-side performance of all three models', width='stretch')
    st.image('fig_model1_linear.png', caption='Model 1: Simple Linear Regression — straight-line fit underfits the curved depreciation', width='stretch')
    st.image('fig_model2_polynomial.png', caption='Model 2: Polynomial Regression (degree 2) — captures curvature but ignores categorical features', width='stretch')
    st.image('fig_model3_feature_importance.png', caption='Model 3: Random Forest feature importance — engine power, age, and manufacturer dominate', width='stretch')
    st.image('fig_model3_pred_vs_actual.png', caption='Random Forest predicted vs actual prices — tight clustering along the diagonal', width='stretch')

    st.markdown(
        "Random Forest is the best model for this problem because it natively handles categorical features (manufacturer, fuel, ownership), "
        "captures non-linear interactions, and is robust to outliers. The Simple Linear baseline fails due to fundamental shape mismatch "
        "with the curved depreciation pattern. Polynomial regression captures the curvature but cannot represent the categorical structure of the data."
    )

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
