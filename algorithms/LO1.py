import os
import pandas as pd
s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
print('From URL:', s)
df = pd.read_csv(s,
header=None,
encoding='utf-8')

print(df.head())
import matplotlib.pyplot as plt
import numpy as np
 # select setosa and versicolor
y = df.iloc[0:100, 4].values
# y = np.where(y == 'Iris-setosa', 0, 1)
# extract sepal length and petal length
X = df.iloc[:100, [1, 3]].values
print(X[:50, 0])
# plot data
plt.scatter(X[:50, 0], X[:50, 1],
color='red', marker='o', label='Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
color='blue', marker='s', label='Versicolor')
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.show()