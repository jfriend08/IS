import sys, scipy
import numpy as np

sys.path.append('../src')
import laplacian, gradient, plotGraph, librosaF
import RecurrenceMatrix as RM

sigmas0 = np.load('./sigmas/modelReal_batchUpdate_Alpha1000_2500/modelReal_batchUpdate_Alpha1000_2500_step0.npy')
sigmas9 = np.load('./sigmas/modelReal_batchUpdate_Alpha1000_2500/modelReal_batchUpdate_Alpha1000_2500_step9.npy')

cqt_med = np.load('./tempArray/cqt_med.npy')
gm0 = RM.feature2GaussianMatrix(cqt_med, sigmas0) #(nSample, nFeature)
L0 = scipy.sparse.csgraph.laplacian(gm0, normed=True)
gm9 = RM.feature2GaussianMatrix(cqt_med, sigmas9) #(nSample, nFeature)
L9 = scipy.sparse.csgraph.laplacian(gm9, normed=True)

m_true = np.load("./tempArray/m_true.npy")
L_true = np.load("./tempArray/L_true.npy")

print (sigmas0==sigmas0.T).all()

print 'max sigma difference ...'
diff = np.absolute(sigmas0-sigmas9)
i,j = np.unravel_index(diff.argmax(), diff.shape)
print np.amax(diff)
print (i,j)

print 'max L0 difference ...'
diff = np.absolute(L_true - L0)
i,j = np.unravel_index(diff.argmax(), diff.shape)
print np.amax(diff)
print (i,j)

print 'max L9 difference ...'
diff = np.absolute(L_true - L9)
i,j = np.unravel_index(diff.argmax(), diff.shape)
print np.amax(diff)
print (i,j)
