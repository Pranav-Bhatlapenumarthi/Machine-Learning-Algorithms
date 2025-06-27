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

Xv = X.values.reshape(-1,1) # reshapes the feature matrix X into a column vector,
h = 0.02 # step size for the mesh grid

# minimum and maximum values for the x and y axes
x_min, x_max = Xv.min(), Xv.max()
y_min, y_max = y.min(), y.max()

# The np.meshgrid function creates a grid of points (xx, yy) covering the feature space
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)

'''
-> xx and yy are 2D arrays
-> ravel() method flattens these arrays to 1D arrays
-> np.c_[] is a way to concatenate arrays columnwise,  resulting in an array where each row is a coordinate pair (x, y) from the grid.
'''

fig = plt.figure(figsize=(8,5))
ax = plt.contourf(xx,yy,z, cmap="afmhot", alpha=0.3) # afmhot is a colour

plt.scatter(X.values[:,0], X.values[:,1], c=y, s=40, alpha=0.9, edgecolors="k")
plt.show()


