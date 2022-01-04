from sklearn.linear_model import LinearRegression

import pandas as pd

linreg = LinearRegression()

titanic_data = pd.read_csv('titanic_train.csv')

titanic_data_no_null =titanic_data.dropna()

titanic_data_no_null.to_csv('new.csv')
