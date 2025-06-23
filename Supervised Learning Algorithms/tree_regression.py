import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree

X = [[0,0], [3,3]] 
y = [0.57, 4]

tree_reg = tree.DecisionTreeRegressor(random_state=42)
tree_reg = tree_reg.fit(X,y)
print(tree_reg.predict([[10.356346, 12.235235]]))

# Creating random dataset 
'''
First, range = np.random.RandomState(1) creates a random number generator with a fixed seed (1), ensuring reproducibility of the
random numbers generated. Next, X = np.sort(5*range.rand(80,1), axis=0) generates 80 random numbers between 0 and 1, scales them to 
the range [0, 5] by multiplying by 5, and sorts them in ascending order. The result, X, is a column vector of 80 sorted values.

The line y = np.sin(X).ravel() computes the sine of each value in X to create the target variable y. 
The .ravel() method flattens the resulting array into a one-dimensional array, which is a common format for regression targets.

Finally, y[::5] += 3*(0.5-range.rand(16)) adds noise to every fifth element of y. The expression range.rand(16) generates 
16 random numbers between 0 and 1. Subtracting these from 0.5 centers the noise around zero, and multiplying by 3 scales the noise. 
'''

range = np.random.RandomState(1)
X = np.sort(5*range.rand(80,1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3*(0.5-range.rand(16))

# Fit regression model
regr_1 = tree.DecisionTreeRegressor(max_depth=2)
regr_2 = tree.DecisionTreeRegressor(max_depth=15, min_samples_leaf=10) # Demonstrates overfitting; min_samples_leaf introduces regularisation
regr_1.fit(X,y)
regr_2.fit(X,y)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
# print(X_test)
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# Plotting the results
plt.figure(figsize=(10,8))
plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=4", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=25", linewidth=2)

plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()