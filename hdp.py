import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import pickle

df = pd.read_csv('dataset.csv')
# print(df.head())
# df = df.dropna("Unnamed: 0", axis=1)

x_df = df.drop("target", axis=1)
y_df = df["target"]


x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3, random_state=42)
# create linear regression model
model = linear_model.LinearRegression()
model.fit(x_train, y_train)
print(model.score(x_train, y_train))

# predinting the test data
pediction_test = model.predict(x_test)
print(y_test, pediction_test)
print('MSE between y_test and predicted value:', np.mean((pediction_test - y_test)**2))

# save the model to disk
pickle.dump(model, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]))