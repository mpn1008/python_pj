import numpy as np
import matplotlib.pyplot as plt

import nnfs
from nnfs.datasets import spiral_data  # See for code: https://gist.github.com/Sentdex/454cb20ec5acf0e76ee8ab8448e6266c

# np.random.seed(0)

nnfs.init()

#
# X = [[1, 2, 3, 2.5],
#      [2, 5, -1, 2],
#      [-1.5, 2.7, 3.3, -0.8]]
#
# inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
#
#
# output = np.maximum(0, inputs)
#
# print(output)
X, y = spiral_data(100, 3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
print(X)
plt.show()
