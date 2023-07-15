import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import pickle

def preprocess_data(data):
    data['date'] = pd.to_datetime(data['date'])
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data['hour'] = data['date'].dt.hour
    data['day_of_week'] = data['date'].dt.dayofweek
    for i in range(1, 25):
        data[f'lag_{i}'] = data['close'].shift(i)
    data = data.dropna()
    return data

def train_and_save_model(X_train, y_train):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    model = GradientBoostingRegressor(random_state=0)
    model.fit(X_train, y_train)
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)
    with open('scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)

def load_model_and_scaler():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return model, scaler

def make_predictions(X, model, scaler):
    X = scaler.transform(X)
    prediction = model.predict(X)
    return prediction

def app():
    st.title("Oil Price Prediction App")


    st.markdown("""
                ## How to use this app
                This application allows you to predict future oil prices using Machine Learning. Here's how to use it:
                1. Prepare your data: This app requires data in a specific format. The data should be in a CSV file with two columns: `date` and `close`. The `date` column should contain the date and time in the format 'DD.MM.YYYY HH:MM:SS.fff' (e.g., '22.02.2022 00:00:00.000'), and the `close` column should contain the closing price of oil at that time. Here's an example of what the data should look like:
                
                | date                  | close  |
                |-----------------------|--------|
                | 22.02.2022 00:00:00.000 | 92.557 |
                | 22.02.2022 01:00:00.000 | 92.927 |
                | 22.02.2022 02:00:00.000 | 92.802 |
                
                You can download this type of data from [Dukascopy](https://www.dukascopy.com/swiss/english/marketwatch/historical/). You'll need to sign up for a free account. The data is hourly in nature and uses GMT time.
                
                2. Upload your data: Use the "Upload CSV" button to upload your data. Once the data is uploaded, you can click the "Retrain model and make predictions" button. The app will retrain the model on your data and make predictions for the next 5, 4, 3, 2, and 1 hours.
                
                Enjoy your forecasting!
                """)

    
    # Load and preprocess the initial training data
    data = pd.read_csv('oil.csv')
    data = preprocess_data(data)
    X = data.drop(['date', 'close'], axis=1)
    y = data['close']
    train_and_save_model(X, y)

    # Allow user to upload new data
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded_file is not None:
        new_data = pd.read_csv(uploaded_file)
        new_data = preprocess_data(new_data)
        X_new = new_data.drop(['date', 'close'], axis=1)
        y_new = new_data['close']
        if st.button("Retrain model and make predictions"):
            train_and_save_model(X_new, y_new)
            st.success("Model successfully retrained on new data!")
            model, scaler = load_model_and_scaler()
            prediction = make_predictions(X_new.iloc[-5:], model, scaler)
            st.write(f"Predicted price for 5 hours ahead: {prediction[-5]}")
            st.write(f"Predicted price for 4 hours ahead: {prediction[-4]}")
            st.write(f"Predicted price for 3 hours ahead: {prediction[-3]}")
            st.write(f"Predicted price for 2 hours ahead: {prediction[-2]}")
            st.write(f"Predicted price for 1 hour ahead: {prediction[-1]}")

if __name__ == "__main__":
    app()

