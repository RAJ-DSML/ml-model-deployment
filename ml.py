import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def score_prediction(score):
    X = pd.read_csv('X.csv')
    y = pd.read_csv('y.csv')
    X = X.values
    y = y.values

    # model = LinearRegression()
    model = LinearRegression()

    # model.fit(X, y)
    model.fit(X, y)

    X_test = np.array(score)
    y_test = X_test.reshape(-1, 1)

    # model prediction
    return model.predict(X_test)

score_prediction(score)
