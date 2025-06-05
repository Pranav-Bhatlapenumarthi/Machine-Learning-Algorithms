import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import sklearn
from sklearn import tree
from sklearn.datasets import load_iris

X = [[0,0], [1,2]]
y = [0,1]

classifier = tree.DecisionTreeClassifier() # instantiation stage
classifier = classifier.fit(X,y) # training stage

print(classifier.predict([[2., 2.]])) #The values are represented as floating point values to ensure numerical consistency
print(classifier.predict_proba([[2., 2.]])) # Gives [[0., 1.]]
# '0.' corresponds to the label '0' in y, indicating that there is no chance that [2., 2.] gives a result of '0'

print(classifier.predict([[0., 0.5]]))
print(classifier.predict_proba([[0., 0.5]]))
print("\n\n------------------------------------------------------\n\n")

# Application of the IRIS dataset
iris = load_iris()
print(iris.data[0:5])
print(iris.feature_names)

X = iris.data[:, 2:]
'''The slicing operation [:, 2:] means "select all rows, but only columns starting from index 2 onward." 
In the context of the iris dataset, columns at index 2 and 3 correspond to "petal length (cm)" and "petal width (cm)"'''
y = iris.target

clf = tree.DecisionTreeClassifier(random_state=42)
clf = clf.fit(X,y)


