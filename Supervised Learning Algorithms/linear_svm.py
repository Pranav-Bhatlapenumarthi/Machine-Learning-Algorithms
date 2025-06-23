from sklearn import svm
from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# We are presently using the IRIS dataset for classification

df = sns.load_dataset("iris")
col = ["petal_length", "petal_width", "species"]
print(df.loc[:, col].head())

col = ["petal_length", "petal_width"]
X = df.loc[:, col] # has only contents with 'col' headings
print(X)

print(df.species.unique())
species_to_num = {"setosa": 0, "versicolor": 1, "virginica":2}
df['tmp'] = df['species'].map(species_to_num)
y = df['tmp']

C = 0.001 # hyperparameter which determines the flexibility of the SVM
# Smaller value of C leads to a wider street but more margin violations
# Higher value of C leads to smaller margin but reduced margin violations

classifier = svm.SVC(kernel="linear", C=C)
classifier.fit(X,y)
print(classifier.predict([[6,2]]))

