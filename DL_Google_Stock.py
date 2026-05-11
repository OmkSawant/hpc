
# =========================================================
# Google Stock Price Prediction using RNN
# =========================================================

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

# =========================================================
# Load Dataset
# =========================================================

# Read CSV file
# skiprows=2 removes unwanted rows (Ticker, NaN rows)
df = pd.read_csv(r"/content/drive/MyDrive/Google_Stock_Price.csv", skiprows=2)

# Rename columns properly
df.columns = [
    'Date',
    'Adj Close',
    'Close',
    'High',
    'Low',
    'Open',
    'Volume'
]

# Display first 5 rows
print(df.head())

# =========================================================
# Select Open Price Column
# =========================================================

# Use Open price for prediction
data = df['Open'].values

# Convert into 2D array
data = data.reshape(-1,1)

# =========================================================
# Feature Scaling
# =========================================================

# Scale data between 0 and 1
scaler = MinMaxScaler(feature_range=(0,1))

data_scaled = scaler.fit_transform(data)

# =========================================================
# Create Training Data
# =========================================================

X_train = []
y_train = []

# Use previous 60 days to predict next day
for i in range(60, len(data_scaled)):

    X_train.append(data_scaled[i-60:i, 0])

    y_train.append(data_scaled[i, 0])

# Convert lists into arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# =========================================================
# Reshape Data for RNN
# =========================================================

# RNN requires 3D input:
# (samples, timesteps, features)

X_train = np.reshape(
    X_train,
    (X_train.shape[0], X_train.shape[1], 1)
)

print("Shape of X_train:", X_train.shape)

# 

# Build RNN Model
# =========================================================

model = Sequential()

# Add RNN Layer
model.add(SimpleRNN(
    units=50,
    activation='tanh',
    input_shape=(X_train.shape[1], 1)
))

# Add Output Layer
model.add(Dense(1))

# =========================================================
# Compile Model
# =========================================================

model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

# =========================================================
# Train Model
# =========================================================

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32
)

# =========================================================
# Predict Stock Prices
# =========================================================

predicted_stock_price = model.predict(X_train)

# Convert scaled values back to original prices
predicted_stock_price = scaler.inverse_transform(
    predicted_stock_price
)

real_stock_price = scaler.inverse_transform(
    y_train.reshape(-1,1)
)

# =========================================================
# Visualize Results
# =========================================================

plt.figure(figsize=(10,5))

# Real Prices
plt.plot(
    real_stock_price,
    color='blue',
    label='Real Google Stock Price'
)

# Predicted Prices
plt.plot(
    predicted_stock_price,
    color='red',
    label='Predicted Google Stock Price'
)

plt.title('Google Stock Price Prediction using RNN')

plt.xlabel('Time')

plt.ylabel('Google Stock Price')

plt.legend()

plt.show()

# Print Sample Predictions
# =========================================================

print("\nSample Predictions:\n")

for i in range(5):

    print("Actual Price    :", real_stock_price[i][0])

    print("Predicted Price :", predicted_stock_price[i][0])

    print("-----------------------------------")

 
