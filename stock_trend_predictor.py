import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Download Apple stock data
data = yf.download('AAPL', start='2015-01-01', end='2024-12-31')
data.columns = data.columns.get_level_values(0)

# Add Tomorrow_Close and Target columns
data['Tomorrow_Close'] = data['Close'].shift(-1)
data['Target'] = (data['Tomorrow_Close'] > data['Close']).astype(int)
data = data[:-1]

# ✅ Add technical indicators BEFORE training
data['MA_3'] = data['Close'].rolling(window=3).mean()
data['MA_7'] = data['Close'].rolling(window=7).mean()
data['Prev_Target'] = data['Target'].shift(1)
data = data.dropna()  # Remove rows with missing values

# ✅ Define features AFTER all columns are ready
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_3', 'MA_7', 'Prev_Target']
X = data[features]
y = data['Target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Results
acc = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {acc:.2f}")

cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(cm)


# Create a new column in the test set to store predictions
X_test = X_test.copy()  # avoid SettingWithCopyWarning
X_test['Actual'] = y_test.values
X_test['Predicted'] = predictions

# Plot actual vs predicted
plt.figure(figsize=(14, 6))
plt.plot(X_test.index, X_test['Actual'], label='Actual Trend (0=Down, 1=Up)', linewidth=2)
plt.plot(X_test.index, X_test['Predicted'], label='Predicted Trend', linestyle='--')
plt.legend()
plt.title("Stock Price Trend Prediction: Actual vs. Predicted")
plt.xlabel("Date")
plt.ylabel("Trend Direction")
plt.grid(True)
plt.show()

# Save the test predictions to a CSV file
X_test.to_csv("stock_trend_predictions.csv")

print("Predictions saved to stock_trend_predictions.csv")
