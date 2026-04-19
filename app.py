import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


model = load_model("Stock_Predictions_Model.h5")

st.header('Stock Market Predictor')

# Market selector
market = st.selectbox(
    "Select Market", 
    ("US Market (NASDAQ/NYSE)", "Indian Market (NSE)", "Indian Market (BSE)")
)

# Set a default ticker based on the market
default_ticker = 'RELIANCE' if 'Indian' in market else 'GOOG'

# Get the raw symbol from the user
raw_symbol = st.text_input('Enter Stock Symbol', default_ticker)

# Automatically format the symbol for yfinance
if market == "Indian Market (NSE)":
    stock = raw_symbol.upper() + ".NS"
elif market == "Indian Market (BSE)":
    stock = raw_symbol.upper() + ".BO"
else:
    stock = raw_symbol.upper()

try:
    ticker_info = yf.Ticker(stock).info
    currency_code = ticker_info.get('currency', 'USD')
except:
    currency_code = 'USD' # Fallback just in case

currency_symbol = '₹' if currency_code == 'INR' else '$'

# Fetch 10 years of daily data up to the current date
data = yf.download(stock, period='10y', interval='1d')

# Safety Check for API blocks
if data.empty:
    st.error(f"No data found for '{stock}'. This is likely due to Yahoo Finance blocking the request or an invalid ticker.")
    st.stop()

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)

st.subheader('Stock Data')
st.write(data)

# Train-test split
data_train = pd.DataFrame(data.Close[0:int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80):])

# Scaling
scaler = MinMaxScaler(feature_range=(0,1))

past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# Moving Averages
ma_50_days = data.Close.rolling(50).mean()
ma_100_days = data.Close.rolling(100).mean()
ma_200_days = data.Close.rolling(200).mean()

# Graph 1
st.subheader('Stock Price vs MA50')
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r', label='MA 50')
plt.plot(data.Close, 'g', label='Closing Price')
plt.legend()
st.pyplot(fig1)

# Graph 2
st.subheader('Stock Price vs MA50 vs MA100')
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r', label='MA 50')
plt.plot(ma_100_days, 'b', label='MA 100')
plt.plot(data.Close, 'g', label='Closing Price')
plt.legend()
st.pyplot(fig2)

# Graph 3
st.subheader('Stock Price vs MA100 vs MA200')
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r', label='MA 100')
plt.plot(ma_200_days, 'b', label='MA 200')
plt.plot(data.Close, 'g', label='Closing Price')
plt.legend()
st.pyplot(fig3)

# Prepare data
x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)

# Predict
predict = model.predict(x)

# Reverse scaling
scale = 1 / scaler.scale_
predict = predict * scale
y = y * scale

## --- EXTRACT THE FINAL PREDICTED VALUE ---
latest_prediction = predict[-1][0]
latest_actual = y[-1]


# --- DISPLAY AS A METRIC ---
st.subheader("Final Prediction")

col1, col2 = st.columns(2)
with col1:
    # Use the dynamic currency_symbol instead of the hardcoded '$'
    st.metric(label="Latest Actual Price", value=f"{currency_symbol}{latest_actual:,.2f}")
with col2:
    st.metric(label="Predicted Next Price", value=f"{currency_symbol}{latest_prediction:,.2f}")
# Final graph
st.subheader('Actual Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(y, 'g', label='Actual Price')
plt.plot(predict, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)