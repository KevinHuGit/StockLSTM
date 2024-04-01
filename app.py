import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import yfinance as yf
import streamlit as st
from pandas_datareader import data as pdr
from keras.models import load_model

yf.pdr_override() # <== that's all it takes :-)
st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')

# download dataframe
data = pdr.get_data_yahoo(user_input, start="2014-01-01", end="2023-12-31")

st.subheader('Data from 2014 - 2024')
st.write(data.describe())

# Visualizations

st.subheader('Closing Price vs. Time Chart')
fig = plt.figure(figsize= (12,6))
plt.plot(data.Close)
st.pyplot(fig)

st.subheader('Closing Price vs. Time Chart with 100 Moving Average')
ma100 = data.Close.rolling(100).mean()
fig = plt.figure(figsize= (12,6))
plt.plot(ma100, 'r')
plt.plot(data.Close)
st.pyplot(fig)

st.subheader('Closing Price vs. Time Chart with 100 and 200 Moving Average')
ma100 = data.Close.rolling(100).mean()
ma200 = data.Close.rolling(200).mean()
fig = plt.figure(figsize= (12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(data.Close)
st.pyplot(fig)

# Split data into training and testing

data_training = pd.DataFrame(data['Close'][0:int(len(data)*0.80)])
data_testing = pd.DataFrame(data['Close'][int(len(data)*0.80): int(len(data))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

# # Splitting data into x_train and y_train
# x_train = []
# y_train = []

# for i in range(100, data_training_array.shape[0]):
#     x_train.append(data_training_array[i-100: i])
#     y_train.append(data_training_array[i, 0])
    
# x_train, y_train = np.array(x_train), np.array(y_train)

# load my model
model = load_model('keras_model.keras')

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index = True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])
    
x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1 / scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Final Graph

st.subheader('Predictive Model vs. Original')

fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Days after first 80% training days')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig2)