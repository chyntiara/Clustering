'''from tsne import bh_sne
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target
X_2d = bh_sne(X)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y)
plt.show()
'''

import numpy as np
import matplotlib.pyplot as plt
from tsne import bh_sne

# load up data
train_file = 'output.csv'
temp_data = np.genfromtxt(train_file, delimiter=',', names=True)
data = temp_data.view((float, len(temp_data.dtype.names)))
x_data = data[:, 2:62]
y_data = data[:, 1]

X_2d = bh_sne(x_data)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_data)
plt.show()
