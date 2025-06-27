import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
import sklearn
import sklearn.neighbors

df = sns.load_dataset("iris")
print(df.head())

print(df.species.unique())

X = df[["petal_length", "petal_width"]]
species_to_num = {"setosa":0,"versicolor":1 ,"virginica": 2}
df['species'] = df['species'].map(species_to_num)
y = df['species']

knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=50)
print(knn.fit(X,y))





