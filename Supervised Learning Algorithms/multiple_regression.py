import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
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

