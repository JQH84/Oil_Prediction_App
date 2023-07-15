# Oil Price Prediction Application

This application uses Machine Learning to predict future oil prices. It is built using Python and Streamlit, and uses a Gradient Boosting Regressor model trained on historical oil price data. 

The application allows users to upload new data, which can be used to both retrain the model and to make predictions for the next 5, 4, 3, 2, and 1 hours.

## How to Use

1. Start the application by running `streamlit run app.py` in your terminal.
2. Navigate to the local URL shown in the terminal to access the application.
3. Use the "Upload CSV" button to upload a CSV file containing your new data. The data should have the same format as the original training data.
4. After uploading new data, click the "Retrain model and make predictions" button. The model will be retrained on the uploaded data, and predictions will be made for the next 5, 4, 3, 2, and 1 hours.

## Technical Details

The application uses a Gradient Boosting Regressor model from the Scikit-Learn library. This model was chosen because it is capable of capturing complex relationships in the data, and tends to perform well on a wide range of datasets.

The model is initially trained on historical oil price data from the 'oil.csv' file. It is then retrained whenever new data is uploaded.

The model uses a variety of features to make its predictions, including the year, month, day, hour, and day of the week, as well as lagged values of the oil price. These features were chosen to capture both the temporal trends and the autocorrelation in the oil price data.

All features are standardized before being fed into the model, using a StandardScaler from Scikit-Learn. This is done to ensure that all features have the same scale, which can help the Gradient Boosting model converge faster.

The performance of the model is evaluated using the Mean Absolute Error (MAE), which gives an idea of how close the model's predictions are to the actual values on average.

## Future Work

Future versions of this application could incorporate additional features, such as oil production data, economic indicators, or geopolitical events, to further improve the accuracy of the predictions. It could also allow users to choose different machine learning models or hyperparameters.
