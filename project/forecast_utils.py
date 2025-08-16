import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# =======================
# Data Cleaning Function
# =======================
def clean_sales_data(df):
    df['sale_date'] = pd.to_datetime(df['sale_date'], errors='coerce')
    df = df.sort_values('sale_date')
    df_grouped = df.groupby(['product_name', 'sale_date'])['quantity_sold'].sum().reset_index()
    all_products = []
    for product in df_grouped['product_name'].unique():
        df_product = df_grouped[df_grouped['product_name'] == product].copy()
        date_range = pd.date_range(df_product['sale_date'].min(), df_product['sale_date'].max())
        df_product = df_product.set_index('sale_date').reindex(date_range).fillna({'product_name': product, 'quantity_sold': 0}).reset_index()
        df_product.rename(columns={'index': 'sale_date'}, inplace=True)
        all_products.append(df_product)
    df_clean = pd.concat(all_products)
    return df_clean

# =======================
# Basic Forecast Function (Moving Average + Linear Regression)
# =======================
def basic_forecast(df_clean, forecast_days=14):
    all_forecasts = []
    warnings = []
    for product in df_clean['product_name'].unique():
        df_product = df_clean[df_clean['product_name'] == product].copy()
        df_product = df_product.sort_values('sale_date')
        window_size = min(7, len(df_product))
        df_product['moving_avg'] = df_product['quantity_sold'].rolling(window=window_size, min_periods=1).mean()
        last_date = df_product['sale_date'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
        last_ma = df_product['moving_avg'].iloc[-1]
        if pd.isna(last_ma):
            last_ma = df_product['quantity_sold'].mean()
        future_ma_forecast = pd.DataFrame({
            'product_name': product,
            'sale_date': future_dates,
            'quantity_sold_ma_forecast': [last_ma] * len(future_dates)
        })
        df_product['days'] = (df_product['sale_date'] - df_product['sale_date'].min()).dt.days
        X = df_product[['days']]
        y = df_product['quantity_sold']
        model = LinearRegression()
        model.fit(X, y)
        future_days = np.array([(date - df_product['sale_date'].min()).days for date in future_dates]).reshape(-1, 1)
        lr_predictions = model.predict(future_days)
        # Allow negative values, but warn if any negative
        if (lr_predictions < 0).any():
            warnings.append(f"⚠️ Warning: Sales for {product} are projected to go negative. Consider reducing stock immediately.")
        future_lr_forecast = pd.DataFrame({
            'product_name': product,
            'sale_date': future_dates,
            'quantity_sold_lr_forecast': lr_predictions
        })
        future_forecast = pd.merge(future_ma_forecast, future_lr_forecast, on=['product_name', 'sale_date'])
        past_data = df_product[['product_name', 'sale_date', 'quantity_sold']].copy()
        past_data['quantity_sold_ma_forecast'] = None
        past_data['quantity_sold_lr_forecast'] = None
        combined = pd.concat([past_data, future_forecast], ignore_index=True)
        all_forecasts.append(combined)
    final_df = pd.concat(all_forecasts, ignore_index=True)
    final_df._warnings = warnings  # Attach warnings for UI
    return final_df

# =======================
# Enhanced Forecast Function (Ensemble, Seasonality, Lag, etc.)
# =======================
def enhanced_forecast(df_clean, forecast_days=14):
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np
    all_forecasts = []
    for product in df_clean['product_name'].unique():
        df_product = df_clean[df_clean['product_name'] == product].copy()
        df_product = df_product.sort_values("sale_date")

        # Feature Engineering
        df_product['day_of_week'] = df_product['sale_date'].dt.dayofweek
        df_product['month'] = df_product['sale_date'].dt.month
        df_product['quarter'] = df_product['sale_date'].dt.quarter
        df_product['day_of_month'] = df_product['sale_date'].dt.day
        df_product['is_weekend'] = df_product['day_of_week'].isin([5, 6]).astype(int)
        df_product['is_month_start'] = (df_product['sale_date'].dt.day <= 5).astype(int)
        df_product['is_month_end'] = (df_product['sale_date'].dt.day >= 25).astype(int)
        df_product['days_from_start'] = (df_product['sale_date'] - df_product['sale_date'].min()).dt.days

        # Lag Features
        df_product['sales_lag_1'] = df_product['quantity_sold'].shift(1)
        df_product['sales_lag_3'] = df_product['quantity_sold'].shift(3)
        df_product['sales_lag_7'] = df_product['quantity_sold'].shift(7)

        # Rolling Statistics
        df_product['rolling_mean_3'] = df_product['quantity_sold'].rolling(window=3, min_periods=1).mean()
        df_product['rolling_mean_7'] = df_product['quantity_sold'].rolling(window=7, min_periods=1).mean()
        df_product['rolling_mean_14'] = df_product['quantity_sold'].rolling(window=14, min_periods=1).mean()
        df_product['rolling_std_7'] = df_product['quantity_sold'].rolling(window=7, min_periods=1).std().fillna(0)
        df_product['rolling_max_7'] = df_product['quantity_sold'].rolling(window=7, min_periods=1).max()
        df_product['rolling_min_7'] = df_product['quantity_sold'].rolling(window=7, min_periods=1).min()

        # Exponential Smoothing
        df_product['exp_smooth_02'] = df_product['quantity_sold'].ewm(alpha=0.2).mean()
        df_product['exp_smooth_05'] = df_product['quantity_sold'].ewm(alpha=0.5).mean()
        df_product['exp_smooth_08'] = df_product['quantity_sold'].ewm(alpha=0.8).mean()

        # Trend and Momentum
        df_product['sales_trend'] = df_product['quantity_sold'].diff().fillna(0)
        df_product['sales_momentum'] = df_product['sales_trend'].rolling(window=3, min_periods=1).mean()
        df_product['sales_acceleration'] = df_product['sales_trend'].diff().fillna(0)

        # Cyclical patterns
        df_product['day_of_week_sin'] = np.sin(2 * np.pi * df_product['day_of_week'] / 7)
        df_product['day_of_week_cos'] = np.cos(2 * np.pi * df_product['day_of_week'] / 7)
        df_product['month_sin'] = np.sin(2 * np.pi * df_product['month'] / 12)
        df_product['month_cos'] = np.cos(2 * np.pi * df_product['month'] / 12)

        feature_cols = [
            'days_from_start', 'day_of_week', 'month', 'quarter', 'day_of_month',
            'is_weekend', 'is_month_start', 'is_month_end',
            'sales_lag_1', 'sales_lag_3', 'sales_lag_7',
            'rolling_mean_3', 'rolling_mean_7', 'rolling_mean_14', 'rolling_std_7',
            'rolling_max_7', 'rolling_min_7',
            'exp_smooth_02', 'exp_smooth_05', 'exp_smooth_08',
            'sales_trend', 'sales_momentum', 'sales_acceleration',
            'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos'
        ]

        modeling_data = df_product.dropna()
        if len(modeling_data) < 10:
            last_exp_smooth = df_product['exp_smooth_05'].iloc[-1]
            future_dates = pd.date_range(start=df_product['sale_date'].max() + pd.Timedelta(days=1), periods=forecast_days)
            simple_forecast = pd.DataFrame({
                'product_name': product,
                'sale_date': future_dates,
                'quantity_sold_enhanced_forecast': [last_exp_smooth] * len(future_dates)
            })
            past_data = df_product[['product_name', 'sale_date', 'quantity_sold']].copy()
            past_data['quantity_sold_enhanced_forecast'] = None
            combined = pd.concat([past_data, simple_forecast], ignore_index=True)
            all_forecasts.append(combined)
            continue

        X = modeling_data[feature_cols]
        y = modeling_data['quantity_sold']
        lr_model = LinearRegression()
        lr_model.fit(X, y)
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        rf_model.fit(X, y)

        last_date = df_product['sale_date'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
        future_predictions = []
        last_row = df_product.iloc[-1].copy()
        for i, future_date in enumerate(future_dates):
            future_features = {
                'days_from_start': last_row['days_from_start'] + i + 1,
                'day_of_week': future_date.dayofweek,
                'month': future_date.month,
                'quarter': future_date.quarter,
                'day_of_month': future_date.day,
                'is_weekend': int(future_date.dayofweek in [5, 6]),
                'is_month_start': int(future_date.day <= 5),
                'is_month_end': int(future_date.day >= 25),
                'day_of_week_sin': np.sin(2 * np.pi * future_date.dayofweek / 7),
                'day_of_week_cos': np.cos(2 * np.pi * future_date.dayofweek / 7),
                'month_sin': np.sin(2 * np.pi * future_date.month / 12),
                'month_cos': np.cos(2 * np.pi * future_date.month / 12),
            }
            if i == 0:
                future_features.update({
                    'sales_lag_1': last_row['quantity_sold'],
                    'sales_lag_3': df_product['quantity_sold'].iloc[-3] if len(df_product) >= 3 else last_row['quantity_sold'],
                    'sales_lag_7': df_product['quantity_sold'].iloc[-7] if len(df_product) >= 7 else last_row['quantity_sold'],
                })
            else:
                future_features.update({
                    'sales_lag_1': future_predictions[i-1]['prediction'] if i >= 1 else last_row['quantity_sold'],
                    'sales_lag_3': future_predictions[i-3]['prediction'] if i >= 3 else last_row['quantity_sold'],
                    'sales_lag_7': future_predictions[i-7]['prediction'] if i >= 7 else last_row['quantity_sold'],
                })
            future_features.update({
                'rolling_mean_3': last_row['rolling_mean_3'],
                'rolling_mean_7': last_row['rolling_mean_7'],
                'rolling_mean_14': last_row['rolling_mean_14'],
                'rolling_std_7': last_row['rolling_std_7'],
                'rolling_max_7': last_row['rolling_max_7'],
                'rolling_min_7': last_row['rolling_min_7'],
                'exp_smooth_02': last_row['exp_smooth_02'],
                'exp_smooth_05': last_row['exp_smooth_05'],
                'exp_smooth_08': last_row['exp_smooth_08'],
                'sales_trend': last_row['sales_trend'],
                'sales_momentum': last_row['sales_momentum'],
                'sales_acceleration': last_row['sales_acceleration'],
            })
            X_future = pd.DataFrame([future_features])[feature_cols]
            lr_pred = lr_model.predict(X_future)[0]
            rf_pred = rf_model.predict(X_future)[0]
            ensemble_pred = 0.3 * lr_pred + 0.7 * rf_pred
            ensemble_pred = max(0, ensemble_pred)
            future_predictions.append({
                'sale_date': future_date,
                'prediction': ensemble_pred,
                'lr_prediction': max(0, lr_pred),
                'rf_prediction': max(0, rf_pred)
            })
        future_forecast = pd.DataFrame({
            'product_name': product,
            'sale_date': [pred['sale_date'] for pred in future_predictions],
            'quantity_sold_enhanced_forecast': [pred['prediction'] for pred in future_predictions],
            'quantity_sold_lr_forecast': [pred['lr_prediction'] for pred in future_predictions],
            'quantity_sold_rf_forecast': [pred['rf_prediction'] for pred in future_predictions],
        })
        past_data = df_product[['product_name', 'sale_date', 'quantity_sold']].copy()
        past_data['quantity_sold_enhanced_forecast'] = None
        past_data['quantity_sold_lr_forecast'] = None
        past_data['quantity_sold_rf_forecast'] = None
        combined = pd.concat([past_data, future_forecast], ignore_index=True)
        all_forecasts.append(combined)
    if all_forecasts:
        final_df = pd.concat(all_forecasts, ignore_index=True)
        return final_df
    else:
        return pd.DataFrame()

# =======================
# Utility to Load Data
# =======================
def load_sales_data(filepath):
    df = pd.read_excel(filepath)
    return df
