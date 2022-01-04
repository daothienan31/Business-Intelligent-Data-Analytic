from sklearn.linear_model import LinearRegression

import pandas as pd

linreg = LinearRegression()

titanic_data = pd.read_csv('titanic_train.csv')

data_with_null = titanic_data.dropna()

data_without_null = data_with_null.dropna()

train_data_x = data_without_null.iloc[:,:6]

train_data_y = data_without_null.iloc[:,6]

linreg.fit(train_data_x, train_data_y)

test_data = data_with_null.iloc[:,:6]
age_predicted['Age'] = pd.DataFrame(linreg.predict(test_data))

data_with_null.Age.fillna(age_predicted.Age,inplace=True)
