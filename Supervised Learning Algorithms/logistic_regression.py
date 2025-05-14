import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression():
    def __init__(self, lr = 0.001, n_iterations = 1000):
        self.lr = lr
        self.n_iterations = n_iterations
        self.bias = None
        self.weights = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iterations):
            linear_predictions = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_predictions)

            # Gradient Descent Computation
            dw = (1/n_samples)*(np.dot(X.T, (predictions - y)))  # we don't explicitly include the summation of the dot product because np..dot() implicitly computes the sum
            db = (1/n_samples)*np.sum(predictions - y) # here there is no dot product, so we need to manually sum up the values
        
            self.weights -= self.lr*dw
            self.bias -= self.lr*db

    def predict(self, X):
        linear_predictions = np.dot(X, self.weights) + self.bias
        predictions = sigmoid(linear_predictions)
        class_predictions = [0 if y <= 0.5 else 1 for y in predictions]
        return class_predictions

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) # Splitting the data into training and testing datasets

classifier = LogisticRegression(lr = 0.01)
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)

def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test)/len(y_test)

acc = accuracy(pred, y_test)
print(acc)

