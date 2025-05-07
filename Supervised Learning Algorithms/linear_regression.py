import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

class LinearRegression:

  def __init__(self, n_iterations = 1000, learning_rate = 0.001): # Constructor
    self.n_iterations = n_iterations
    self.learning_rate = learning_rate
    self.weights = None
    self.bias = None

  # Used to train the model
  def fit(self, X, y):
    samples, features = X.shape #gives the dimensions of the array in the form of a tuple -> (total rows, total columns)
    self.weights = np.zeros(features)
    self.bias = 0

    for i in range(self.n_iterations):
      y_prediction = np.dot(X, self.weights) + self.bias # y = wX + b

      # Gradient Descent Computation
      dw = (1/samples)*(np.dot(X.T, (y_prediction - y)))  # we don't explicitly include the summation of the dot product because np..dot() implicitly computes the sum
      db = (1/samples)*np.sum(y_prediction - y) # here there is no dot product, so we need to manually sum up the values

      # Updating the weights and biases
      self.weights -= self.learning_rate*dw
      self.bias -= self.learning_rate*db

  def predict(self, X):
    y_prediction = np.dot(X, self.weights) + self.bias
    return y_prediction

X,y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

fig = plt.figure(figsize=(8,6))
plt.scatter(X[:,0], y, color="b", marker="o", s=30)
plt.show()

reg = LinearRegression(learning_rate = 0.1) # for better prediction
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)

# Mean Square Error Computation
def MSE(y_test, predictions):
  return np.mean((y_test - predictions)**2)

mse = MSE(y_test, predictions)
print(mse)

# Plotting the regression line
predicition_line = reg.predict(X)
cmap = plt.get_cmap('viridis')

fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, predicition_line, color='black', linewidth=2, label='Prediction')
plt.show()
