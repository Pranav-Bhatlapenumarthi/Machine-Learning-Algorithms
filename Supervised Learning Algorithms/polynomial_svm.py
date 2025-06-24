'''
This is used to transform linearly inseparable classes into 2-D separable class.
Polynomial SVMs use a polynomial kernel to achieve the same.
We use the IRIS dataset to demonstrate the functionality of the SVM classifier.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = sns.load_dataset("iris")
col = ["petal_length", "petal_width"]
X = df.loc[:, col]
species_to_num = {"setosa": 0, "versicolor": 1,  "virginica": 2 }
df['tmp'] = df['species'].map(species_to_num)
y = df['tmp']

X_train, X_std_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

sc = StandardScaler()
X_std_test = sc.fit_transform(X_train)

C = 1.0
clf = svm.SVC(kernel="poly", C=C, degree=10, gamma="auto")
clf.fit(X_std_test, y_train)

print(clf.predict([[6,2]]))

'''
The results may appear similar to the linear_svm.py program due to the similarity in the datasets.
But they show significant variation in F1_score, Accuracy and Precision metrics

To improve the performance, hyperparameter tuning can be employed. Here, "degree" and "C" hyperparameters can be tuned to improve the performance
'''