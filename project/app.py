import streamlit as st
import pandas as pd
import os
from forecast_utils import load_sales_data, clean_sales_data, basic_forecast, enhanced_forecast

st.set_page_config(page_title="Inventory Forecast Dashboard", layout="wide")
st.title("📦 Inventory Analysis & Forecasting Dashboard")

# Sidebar for file upload and options
st.sidebar.header("Upload Sales Data")
uploaded_file = st.sidebar.file_uploader("Choose an Excel file", type=["xlsx"])

forecast_type = st.sidebar.selectbox("Forecast Type", ["Basic (Moving Average + Linear Regression)", "Enhanced (Ensemble, Seasonality, Lag)"])
forecast_days = st.sidebar.slider("Forecast Days", min_value=7, max_value=30, value=14)

if uploaded_file:
    # Load and clean data
    df = load_sales_data(uploaded_file)
    st.subheader("Raw Data Preview")
    df_display = df.copy()
    if 'sale_date' in df_display.columns:
        df_display['sale_date'] = pd.to_datetime(df_display['sale_date'], errors='coerce').dt.date
    st.dataframe(df_display)

    df_clean = clean_sales_data(df)
    st.subheader("Cleaned Data Preview")
    df_clean_display = df_clean.copy()
    if 'sale_date' in df_clean_display.columns:
        df_clean_display['sale_date'] = pd.to_datetime(df_clean_display['sale_date'], errors='coerce').dt.date
    st.dataframe(df_clean_display)

    # Forecast
    st.header("Forecast Results")
    if forecast_type.startswith("Basic"):
        forecast_df = basic_forecast(df_clean, forecast_days=forecast_days)
        st.success("Basic forecast generated.")
        if hasattr(forecast_df, '_warnings') and forecast_df._warnings:
            for w in forecast_df._warnings:
                st.warning(w)
    else:
        forecast_df = enhanced_forecast(df_clean, forecast_days=forecast_days)
        st.success("Enhanced forecast generated.")

    # Show forecast table for selected product only
    forecast_products = forecast_df['product_name'].dropna().unique()
    selected_forecast_product = st.selectbox("Select Product for Forecast Table", forecast_products)
    st.subheader(f"Forecast for {selected_forecast_product}")
    forecast_cols = [col for col in forecast_df.columns if col != 'quantity_sold' and col != 'product_name']
    product_forecast = forecast_df[forecast_df['product_name'] == selected_forecast_product]

    # Only show rows after last available date in cleaned data
    last_actual_date = df_clean[df_clean['product_name'] == selected_forecast_product]['sale_date'].max()
    future_rows = product_forecast['sale_date'] > last_actual_date

    # Reset index so numbering starts from 1
    forecast_display = product_forecast.loc[future_rows, forecast_cols].reset_index(drop=True)
    forecast_display.index = forecast_display.index + 1
    if 'sale_date' in forecast_display.columns:
        forecast_display['sale_date'] = pd.to_datetime(forecast_display['sale_date'], errors='coerce').dt.date

    st.dataframe(forecast_display)

    # Download forecast (fix: use BytesIO for Excel export)
    import io
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        forecast_df.to_excel(writer, index=False)
    excel_data = output.getvalue()
    st.download_button(
        label="Download Forecast as Excel",
        data=excel_data,
        file_name="forecast_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # Product selection for visualization
    products = forecast_df['product_name'].dropna().unique()
    selected_product = st.selectbox("Select Product for Visualization", products)
    df_product = forecast_df[forecast_df['product_name'] == selected_product]

    # Plot sales and forecast
    st.subheader(f"Sales & Forecast for {selected_product}")
    if forecast_type.startswith("Basic"):
        chart_cols = ['quantity_sold', 'quantity_sold_ma_forecast', 'quantity_sold_lr_forecast']
    else:
        chart_cols = ['quantity_sold', 'quantity_sold_enhanced_forecast', 'quantity_sold_lr_forecast', 'quantity_sold_rf_forecast']
    available_cols = [col for col in chart_cols if col in df_product.columns]
    st.line_chart(df_product.set_index('sale_date')[available_cols].fillna(0))

    # Pie chart for product contribution
    st.subheader("Product Contribution to Total Sales")
    product_totals = df_clean.groupby('product_name')['quantity_sold'].sum()
    total_inventory = product_totals.sum()
    # Avoid division by zero
    if total_inventory == 0:
        pie_labels = [f"{product} (0.0%)" for product in product_totals.index]
        pie_values = [0 for _ in product_totals.index]
    else:
        pie_labels = [f"{product} ({(value/total_inventory)*100:.1f}%)" for product, value in product_totals.items()]
        pie_values = list(product_totals.values)
    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Pie(labels=pie_labels, values=pie_values, textinfo='label', hole=0.3)])
    fig.update_layout(title="Product Contribution to Total Sales", title_x=0.5)
    st.plotly_chart(fig)
else:
    st.info("Please upload a sales data Excel file to begin analysis.")
