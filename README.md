# 📈 Stock Price Trend Prediction using Machine Learning

This project uses machine learning to predict whether a stock's price will go **up or down** the next day based on historical stock data. The model is trained on Apple Inc. (AAPL) stock prices using features like open, high, low, close, volume, moving averages, and previous trend direction.

---

## 🧠 Project Overview

- **Goal**: Predict the next day's trend — 📉 (0 = down or same), 📈 (1 = up)
- **Model Used**: Random Forest Classifier
- **Dataset**: Real-time financial data from Yahoo Finance via `yfinance`
- **Stock Analyzed**: Apple Inc. (AAPL)

---

## 📦 Tech Stack

- Python
- yfinance (data scraping)
- Pandas, NumPy (data manipulation)
- scikit-learn (machine learning)
- Matplotlib (visualization)

---

## 🔍 Features Used

- Open, High, Low, Close, Volume
- 3-day and 7-day Moving Averages
- Previous day’s price movement (Target lag)

---

## 📊 Results

- **Model Accuracy**: ~47%
- Confusion Matrix shows more accurate predictions for “price down” cases
- Visual comparison of actual vs. predicted trends
- CSV file of test predictions saved as `stock_trend_predictions.csv`

---

## 📁 Output Files

- `stock_trend_predictions.csv`: Contains actual vs. predicted values on the test set
- Plot of trend predictions shown during execution

---

## 📌 Limitations

- Only technical price indicators are used (no news/sentiment data)
- Model does not consider long-term historical patterns or deep time series methods
- Performance may vary significantly across different stocks

---

## ✅ Future Improvements

- Try LSTM (Recurrent Neural Networks) for time series modeling
- Use news sentiment or social media data
- Add more technical indicators (RSI, MACD, Bollinger Bands)
- Create a live Streamlit dashboard

---

## 📬 Author

Abdullah Bin Zubair  
Senior Bachelors Student | Machine Learning & AI Enthusiast  
[GitHub Profile](https://github.com/realabdullahbinzubair)
