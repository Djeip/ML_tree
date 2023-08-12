import numpy as np
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

data = pd.read_csv(r'C:\Users\Lenovo\Documents\golf_players.csv')
data.head()

X = data.drop('Golf Players', axis=1)
y = data['Golf Players']

enc = OneHotEncoder()
X = enc.fit_transform(X)


class GradientBoostingCustom(BaseEstimator):
    def __init__(self, n_estimators, max_depth=3, random_state=23):
        self.y = None
        self.X = None
        self.a = None
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y, lr=0.01):
        self.a = []
        self.X = X
        self.y = y

        y_pred = np.full(len(y), np.mean(y))
        self.a.append(y_pred)

        for _ in range(self.n_estimators):
            resid = y - self.a[-1]
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
            tree.fit(X, resid)
            b = tree.predict(X).reshape([X.shape[0]])
            y_pred = self.a[-1] + lr * b
            self.a.append(y_pred)
            self.trees.append(tree)

    def predict(self, X, a=None, b=None):
        if a is None:
            a = self.a[-2]
        if b is None:
            b = self.trees[-1]
        return self.a[-2] + b.predict(X)

    @staticmethod
    def mse(yt, y):
        error = np.sum(np.power(yt - y, 2)) / 2
        return error

    def iterplot(self, X):
        loss = []
        for obj in zip(self.a, self.trees):
            y_pred = self.predict(X, obj[0], obj[1])
            loss.append(self.mse(y, y_pred))
        plt.xlabel('i')
        plt.ylabel('mse')
        plt.plot(list(range(1, self.n_estimators+1)), loss)
        plt.show()


model = GradientBoostingCustom(3)

model.fit(X,y)
model.iterplot(X)