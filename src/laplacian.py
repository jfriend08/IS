import numpy as np
import scipy

def getNormLaplacian(matrix, m):
  L, d = scipy.sparse.csgraph.laplacian(matrix, normed=True, return_diag=True)
  eigVals, eigVecs = np.linalg.eig(L)
  vals_s = sorted([[i, v] for i, v in enumerate(eigVals)], key=lambda x: x[1])
  Y = eigVecs[:,[vals_s[i][0] for i in xrange(m)]]
  return Y
