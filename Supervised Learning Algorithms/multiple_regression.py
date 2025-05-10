import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
import statsmodels.api as sm
import statsmodels.formula.api as smf


california_data = fetch_california_housing()
df = pd.DataFrame(california_data.data, columns = california_data.feature_names)

X = df # Independent variables
y = california_data.target # Dependent variables (median values)

# To add the bias constant (independent variable) to the other independent variables
X_const = sm.add_constant(X)
pd.DataFrame(X_const)

model = sm.OLS(y, X_const) # Ordinary Least Squares (OLS) is used for finding the best-fitting model
lr = model.fit()
print(lr.summary()) # Statistical analysis of the data

# To avoid multicollinearity, we use a correlation matrix to spot and modify the specific values

pd.options.display.float_format = '{:,.2f}'.format
corr = X.corr()
print(corr)

corr[np.abs(corr) < 0.6] = 0
print(corr)

plt.figure(figsize=(10,10))
sns.heatmap(corr, annot=True, cmap='YlGnBu')
plt.show()

from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(y, lr.fittedvalues))
print(f"RMSE: {rmse:.3f}")
print(f"R-squared: {lr.rsquared:.3f}")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_const, y, test_size=0.2, random_state=0)

model_split = sm.OLS(y_train, X_train).fit()
y_pred = model_split.predict(X_test)

test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {test_rmse:.3f}")
