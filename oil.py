import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

print("Loading data...")
data = pd.read_csv('oil.csv')

print("Preprocessing data...")
data['date'] = pd.to_datetime(data['date'])
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['hour'] = data['date'].dt.hour
data['day_of_week'] = data['date'].dt.dayofweek
for i in range(1, 25):
    data[f'lag_{i}'] = data['close'].shift(i)
data = data.dropna()

print("Splitting data into training and test sets...")
X = data.drop(['date', 'close'], axis=1)
y = data['close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print("Standardizing the features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Training the model...")
gb_model = GradientBoostingRegressor(random_state=0)
gb_model.fit(X_train, y_train)

print("Making predictions...")
y_pred = gb_model.predict(X_test)

# Calculate the mean absolute error of the predictions
mae = mean_absolute_error(y_test, y_pred)

# Save the predictions and the actual values to a CSV file
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results.to_csv('results.csv', index=False)

print(f'Mean Absolute Error: {mae}')
