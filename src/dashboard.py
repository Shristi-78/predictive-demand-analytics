import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

st.title("Hybrid Prophet + XGBoost: Multi-Metal Price Forecast")

metals = ['lithium', 'copper', 'nickel', 'aluminum', 'cobalt']
results = {}

# 1. Loop over metals and run hybrid model
for metal in metals:
    # Generate synthetic data
    np.random.seed(hash(metal) % 2**32)
    dates = pd.date_range(start='2022-01-01', periods=36, freq='ME')
    base_price = {
        'lithium': 18000, 'copper': 9000, 'nickel': 20000, 'aluminum': 2500, 'cobalt': 35000
    }[metal]
    price = base_price + np.cumsum(np.random.normal(300, 200, 36))
    df = pd.DataFrame({
        'date': dates,
        'price_usd_per_ton': price,
        'policy_flag': np.random.choice([0, 1], 36, p=[0.85, 0.15]),
        'trade_dispute_flag': np.random.choice([0, 1], 36, p=[0.9, 0.1]),
        'natural_disaster_flag': np.random.choice([0, 1], 36, p=[0.95, 0.05]),
        'tech_innovation_flag': np.random.choice([0, 1], 36, p=[0.9, 0.1]),
        'gdp_growth': np.random.normal(6, 1, 36),
        'ore_grade_index': np.random.normal(0.8, 0.05, 36),
        'inventory_level': np.random.normal(10000, 2000, 36),
        'sector_demand_index': np.random.normal(1.0, 0.2, 36),
        'sentiment': np.random.normal(0, 1, 36)
    })

    # Prophet
    prophet_df = df.rename(columns={'date': 'ds', 'price_usd_per_ton': 'y'})[['ds', 'y']]
    test_size = int(0.2 * len(prophet_df))
    train_df = prophet_df.iloc[:-test_size]
    model_prophet = Prophet()
    model_prophet.fit(train_df[['ds', 'y']])
    forecast = model_prophet.predict(prophet_df[['ds']])
    df['prophet_pred'] = forecast['yhat']
    df['residual'] = df['price_usd_per_ton'] - df['prophet_pred']

    # XGBoost on residuals
    features = ['policy_flag', 'trade_dispute_flag', 'natural_disaster_flag', 
                'tech_innovation_flag', 'gdp_growth', 'ore_grade_index', 
                'inventory_level', 'sector_demand_index', 'sentiment']
    X = df[features]
    y_resid = df['residual']
    X_train, X_test, y_train, y_test = train_test_split(X, y_resid, test_size=0.2, shuffle=False)
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3)
    xgb_model.fit(X_train, y_train)
    resid_pred = xgb_model.predict(X_test)

    # Final hybrid prediction
    prophet_pred_test = df['prophet_pred'].iloc[X_test.index]
    hybrid_pred = prophet_pred_test + resid_pred
    actual = df['price_usd_per_ton'].iloc[X_test.index]
    mae = mean_absolute_error(actual, hybrid_pred)

    # Future forecast
    future = model_prophet.make_future_dataframe(periods=6, freq='M')
    forecast_future = model_prophet.predict(future)
    future_price = float(forecast_future['yhat'].iloc[-1])

    # Store results
    results[metal] = {
        'future_price': future_price,
        'mae': mae,
        'hybrid_pred': hybrid_pred,
        'actual': actual,
        'dates': df['date'].iloc[X_test.index],
        'forecast_future': forecast_future
    }

# 2. Display table of future prices
future_prices = {metal: results[metal]['future_price'] for metal in metals}
st.write("## Predicted Future Prices (USD/ton)")
st.table(future_prices)

# 3. Highlight the highest value metal
max_metal = max(future_prices, key=future_prices.get)
st.success(f"The predicted highest value metal is: **{max_metal.capitalize()}** at ${future_prices[max_metal]:,.2f} per ton")

# 4. Let user select a metal to view details
selected_metal = st.selectbox("Select a metal to view details:", metals)
res = results[selected_metal]

# 5. Plot actual vs hybrid predicted (test set)
st.subheader(f"{selected_metal.capitalize()}: Actual vs Hybrid Predicted (Test Set)")
fig, ax = plt.subplots()
ax.plot(res['dates'], res['actual'], label='Actual')
ax.plot(res['dates'], res['hybrid_pred'], label='Hybrid Predicted')
ax.legend()
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD/ton)')
st.pyplot(fig)

# 6. Show Prophet forecast table for next 6 months
st.write(f"Prophet Forecast for {selected_metal.capitalize()} (next 6 months):")
st.dataframe(res['forecast_future'][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(6)) 