import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

mnist = fetch_openml(name='mnist_784')
print(len(mnist.data)) #printing the number of rows

X, y = mnist['data'], mnist['target']
y = y.astype(int)

def vizualise(n):
    plt.imshow(X.iloc[n].values.reshape(28,28), cmap='gray')
    plt.show()
    return

print(y == 4)
print(np.where(y == 4)) #printing the index of the rows where the target is 4

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) #splitting the data into training and testing sets

# We setup the conditional data for binary classification here
y_train_0 = np.array(y_train == 0)
y_test_0 = (y_test == 0)
print(type(y_train_0))

X_train2 = np.array(X_train)

# Using Stochastic Gradient Descent (SGD) Classifier
clf = SGDClassifier(random_state=0)
clf.fit(X_train2, y_train_0) #fitting the model

print(clf.predict(X.iloc[1234].values.reshape(1, -1))) #predicting the target of the 1234th row
vizualise(1234)

print(clf.predict(X.iloc[12].values.reshape(1, -1))) #predicting the target of the 12th row
vizualise(12)

print(clf.predict(X.iloc[1000].values.reshape(1, -1))) #predicting the target of the 1000th row
vizualise(1000)