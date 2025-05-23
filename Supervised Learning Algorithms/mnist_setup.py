import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns

mnist = fetch_openml(name='mnist_784')
print(len(mnist.data)) #printing the number of rows

X, y = mnist['data'], mnist['target']
print(X) #printing the data
print(y) #printing the target

def vizualise(n):
    plt.imshow(X.iloc[n].values.reshape(28,28), cmap='gray')
    plt.show()
    return

print(y[1000]) #printing the target of the 1000th row
vizualise(1000) #visualising the 1000th row

