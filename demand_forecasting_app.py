import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from all four CSV files
data1 = pd.read_csv('Transactional_data_retail_01.csv')  # Adjust filename
data2 = pd.read_csv('Transactional_data_retail_02.csv')  # Adjust filename
data3 = pd.read_csv('CustomerDemographics.csv')  # Adjust filename
data4 = pd.read_csv('ProductInfo.csv')  # Adjust filename

# Combine the transactional data
data = pd.concat([data1, data2], ignore_index=True)

# Convert InvoiceDate column to datetime
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], format='%d-%m-%Y', errors='coerce')

# Group by StockCode and sum the 'Quantity'
product_data = data.groupby('StockCode')['Quantity'].sum().reset_index()
top_products = product_data.nlargest(10, 'Quantity')['StockCode'].tolist()

# Create a daily demand dataframe
daily_data = data.groupby(['InvoiceDate', 'StockCode'])['Quantity'].sum().reset_index()

st.title('Demand Forecasting App')

# Dropdown for selecting top 10 products
selected_stock_code = st.selectbox('Select a Stock Code:', top_products)

# Filter data for selected stock code
filtered_data = daily_data[daily_data['StockCode'] == selected_stock_code]

# Display filtered data
st.subheader('Dataset for Selected Product')
st.write(filtered_data.head())

# EDA section
st.subheader('Data Analysis')
st.line_chart(filtered_data.set_index('InvoiceDate')['Quantity'])

# Train ARIMA model for the selected stock code
st.subheader('Train ARIMA Model')
if st.button('Train Model'):
    model = ARIMA(filtered_data['Quantity'], order=(1, 1, 1))
    model_fit = model.fit()
    st.write(model_fit.summary())
    
    # Forecasting
    forecast = model_fit.forecast(steps=15)
    
    # Create a combined DataFrame for plotting
    forecast_index = pd.date_range(start=filtered_data['InvoiceDate'].iloc[-1] + pd.Timedelta(days=1), periods=15)
    forecast_df = pd.DataFrame({'InvoiceDate': forecast_index, 'Forecast': forecast})

    # Historical and Forecast Plot
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_data['InvoiceDate'], filtered_data['Quantity'], label='Actual Demand', color='blue')
    plt.plot(forecast_df['InvoiceDate'], forecast_df['Forecast'], label='Forecast', color='orange')
    plt.xlabel('Invoice Date')
    plt.ylabel('Quantity')
    plt.title(f'Demand Forecast for {selected_stock_code}')
    plt.legend()
    st.pyplot(plt)

    # Error Analysis
train_size = int(len(filtered_data) * 0.8)
train, test = filtered_data['Quantity'][:train_size], filtered_data['Quantity'][train_size:]

# Calculate errors
train_forecast = model_fit.predict(start=train.index[0], end=train.index[-1])
test_forecast = model_fit.predict(start=test.index[0], end=test.index[-1])

train_error = train - train_forecast
test_error = test - test_forecast

# Error Histogram with Seaborn
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.histplot(train_error, bins=20, color='blue', kde=True)
plt.title('Training Error Distribution')
plt.xlabel('Error')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.histplot(test_error, bins=20, color='orange', kde=True)
plt.title('Testing Error Distribution')
plt.xlabel('Error')

st.pyplot(plt)