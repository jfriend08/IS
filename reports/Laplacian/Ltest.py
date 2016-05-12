import numpy as np
import sys, scipy, os
import scipy.sparse
from scipy.sparse import isspmatrix

sys.path.append('../src')
import laplacian
import RecurrenceMatrix as RM
import similarityMatrix as SM

def _setdiag_dense(A, d):
  A.flat[::len(d)+1] = d

def _laplacian_dense(graph, normed=False, axis=0):
  m = np.array(graph)
  np.fill_diagonal(m, 0)
  w = m.sum(axis=axis)
  print w
  print w[:, np.newaxis].shape
  if normed:
    w = np.sqrt(w)
    isolated_node_mask = (w == 0)
    w[isolated_node_mask] = 1
    print m
    m /= w
    m /= w[:, np.newaxis]
    m *= -1
    _setdiag_dense(m, 1 - isolated_node_mask)
  else:
    m *= -1
    _setdiag_dense(m, w)
  return m, w


o3 = SM.getOffDiagMatrixIII(3,5)
print o3.shape
print "o3 is symmetric: ", (o3 == np.transpose(o3)).all()
print "isspmatrix(o3): ", isspmatrix(o3)


L1, d1 = _laplacian_dense(o3, True)
print L1

L2, d2 = scipy.sparse.csgraph.laplacian(o3, normed=True, return_diag=True)
print L2

print (L1 == L2).all()



