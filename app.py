import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load model
model = load_model('Stock Predictions Model.keras')
# User input
stock = input("Enter Stock Symbol (e.g., GOOG, AAPL): ")

start = '2012-01-01'
end = '2022-12-31'

# Download data
data = yf.download(stock, start, end)

print("\nStock Data Preview:\n")
print(data.head())

# Split data
data_train = pd.DataFrame(data.Close[0:int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80):])

# Scaling
scaler = MinMaxScaler(feature_range=(0,1))

past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# Moving averages
ma_50 = data.Close.rolling(50).mean()
ma_100 = data.Close.rolling(100).mean()
ma_200 = data.Close.rolling(200).mean()

# Plot 1
plt.figure(figsize=(8,6))
plt.plot(data.Close, label='Original Price')
plt.plot(ma_50, label='MA50')
plt.legend()
plt.title("Price vs MA50")
plt.show()

# Plot 2
plt.figure(figsize=(8,6))
plt.plot(data.Close, label='Original Price')
plt.plot(ma_50, label='MA50')
plt.plot(ma_100, label='MA100')
plt.legend()
plt.title("Price vs MA50 vs MA100")
plt.show()

# Plot 3
plt.figure(figsize=(8,6))
plt.plot(data.Close, label='Original Price')
plt.plot(ma_100, label='MA100')
plt.plot(ma_200, label='MA200')
plt.legend()
plt.title("Price vs MA100 vs MA200")
plt.show()

# Prepare test data
x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x, y = np.array(x), np.array(y)

# Prediction
predict = model.predict(x)

scale = 1 / scaler.scale_

predict = predict * scale
y = y * scale

# Plot prediction
plt.figure(figsize=(8,6))
plt.plot(y, label='Original Price')
plt.plot(predict, label='Predicted Price')
plt.legend()
plt.title("Original vs Predicted Price")
plt.show()