import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sphinx.cmd.quickstart import nonempty

df = pd.read_csv("insurance.csv")

df[df.select_dtypes('int64').columns] = df.select_dtypes('int64').astype('float64')

df = df[['age', 'bmi', 'children', 'charges']]
#print(df)

#X = df[['age', 'bmi', 'children']].values
X = df[['age', 'bmi']].values
y = df['charges'].values.reshape(-1, 1)

reg = LinearRegression().fit(X, y)
y_pred = reg.predict(X)
rmse = root_mean_squared_error(y,y_pred)

class LinReg:
    def initialize(self):
        self.coef_ = None
        self.intercept_ = None
    def fit(self, X, y):
        M = np.column_stack((np.ones(X.shape[0]), X))
        beta = np.linalg.inv(M.T @ M) @ M.T @ y
        self.intercept_ = beta[0]
        self.coef_ = beta[1:]
        return self
    def predict(self, X):
        return X @ self.coef_ + self.intercept_
    def score(self, X, y):
        return 1.0 - np.sum((self.predict(X)-y) ** 2.0)/np.sum((y-np.average(y)) ** 2.0)
    def RMSE(self, X, y):
        return np.sqrt(np.average((self.predict(X)-y) ** 2.0))

reg_mine = LinReg().fit(X, y)
y_pred_mine = reg_mine.predict(X)
rmse_mine = reg_mine.RMSE(X, y)

print("--------    SKLearn    --------")
print("R^2 Score: ", reg.score(X, y))
print("RMSE:      ", rmse)
print()
print("--------     Mine      --------")
print("R^2 Score: ", reg_mine.score(X, y))
print("RMSE:      ", rmse_mine)
#print("Coefficients:", reg.coef_)
#print("Intercept:", reg.intercept_)
