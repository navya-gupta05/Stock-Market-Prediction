# 📈 Stock Market Prediction using LSTM (Keras)

This project is a **beginner-friendly stock market prediction system** built using **Python, Machine Learning, and Deep Learning (LSTM)**. It predicts stock prices based on historical data and visualizes trends using moving averages.

---

## Features

*  Fetches real-time stock data using `yfinance`
*  Uses a trained **Keras LSTM model** for prediction
*  Visualizes:

  * Price vs MA50
  * Price vs MA50 & MA100
  * Price vs MA100 & MA200
*  Predicts future stock trends
*  Compares actual vs predicted prices

---

## Tech Stack

* Python 
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* TensorFlow / Keras
* yFinance API

---

## Project Structure

```
├── app.py
├── Stock_Market_Prediction_Model_Creation.ipynb
├── Stock Predictions Model.keras
├── README.md
```

---

## Installation

1. Clone the repository:

```
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

2. Install dependencies:

```
pip install numpy pandas matplotlib scikit-learn tensorflow yfinance
```

---

## How to Run

Run the Python script:

```
python app.py
```

Then enter a stock symbol:

```
Enter Stock Symbol (e.g., GOOG, AAPL):
```

---

## Example Output

* Moving average graphs
* Stock price visualization
* Original vs Predicted price comparison

---

## Model Details

* Model Type: LSTM Neural Network
* Framework: TensorFlow / Keras
* Input: Last 100 days stock data
* Output: Predicted stock price

---

## ⚠️ Disclaimer

This project is for **educational purposes only**.
Stock market predictions are not guaranteed and should not be used for real financial decisions.


Navya Gupta
