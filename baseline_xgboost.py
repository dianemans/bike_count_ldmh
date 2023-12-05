import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

data = pd.read_parquet('train.parquet')
X_test = pd.read_parquet('final_test.parquet')

data['date_year'] = data['date'].dt.year
data['date_month'] = data['date'].dt.month
data['date_day'] = data['date'].dt.day
data['CI_year'] = data['counter_installation_date'].dt.year
data['CI_month'] = data['counter_installation_date'].dt.month
data['CI_day'] = data['counter_installation_date'].dt.day

data.drop(['date', 'counter_installation_date'], axis=1, inplace=True)

X_test['date_year'] = X_test['date'].dt.year
X_test['date_month'] = X_test['date'].dt.month
X_test['date_day'] = X_test['date'].dt.day
X_test['CI_year'] = X_test['counter_installation_date'].dt.year
X_test['CI_month'] = X_test['counter_installation_date'].dt.month
X_test['CI_day'] = X_test['counter_installation_date'].dt.day

X_test.drop(['date', 'counter_installation_date'], axis=1, inplace=True)

label_encoders = {}
categorical_columns = ['counter_id', 'counter_name', 'site_name', 'coordinates', 'counter_technical_id']

# Fit a label encoder for each column and store it in the dictionary
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Use the stored label encoders to transform the test set
for col in categorical_columns:
    # Use the encoder for the column
    X_test[col] = label_encoders[col].transform(X_test[col])

X_train = data.drop(['log_bike_count', 'bike_count'], axis=1)
y_train = data['log_bike_count']

from xgboost import XGBRegressor
model = XGBRegressor(objective='reg:squarederror', enable_categorical=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

submission = pd.DataFrame({'Id': X_test.index,
                           'log_bike_count': y_pred})

submission.to_csv('submission.csv', index=False)