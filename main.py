from time import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

from SQuaD_MDS import run_SQuaD_MDS
from hybrid import run_hybrid


def get_satellite():
    XY = np.genfromtxt('satellite.csv', delimiter=";", skip_header=1)
    Y = XY[:, -1] - 1
    X = XY[:, :-1]
    return zscore(X, axis=0), Y

X, Y = get_satellite()


print("running the algorithms with the python code (BH-tSNE for the hybrid verison, carefull if N is big)")
X_mds = run_SQuaD_MDS(X, {'in python':True})
X_hybrid = run_hybrid(X, {'in python':True})

plt.scatter(X_mds[:, 0], X_mds[:, 1], c=Y)
plt.show()
plt.scatter(X_hybrid[:, 0], X_hybrid[:, 1], c=Y)
plt.show()


print("running the algorithms with the C++ code (same as in the paper, using FIt-SNE for a O(N) time complexity)")
X_mds = run_SQuaD_MDS(X, {'in python':False})
X_hybrid = run_hybrid(X, {'in python':False})

plt.scatter(X_mds[:, 0], X_mds[:, 1], c=Y)
plt.show()
plt.scatter(X_hybrid[:, 0], X_hybrid[:, 1], c=Y)
plt.show()
