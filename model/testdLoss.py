import sys, scipy
import numpy as np

sys.path.append('../src')
import laplacian, gradient, plotGraph, librosaF
import RecurrenceMatrix as RM


cqt_med = np.load('./tempArray/cqt_med.npy')
# sigmas = np.load('./sigmas/modelReal_batchUpdate_Alpha1000_2500/modelReal_batchUpdate_Alpha1000_2500_step0.npy')
sigmas = np.random.rand(cqt_med.shape[0], cqt_med.shape[0])
sigmas = ((sigmas + sigmas.T)/2)
gm = RM.feature2GaussianMatrix(cqt_med, sigmas) #(nSample, nFeature)

delta = 1e-7
pos1, pos2 = 1, 300

L = scipy.sparse.csgraph.laplacian(gm, normed=True)
L_true = np.load("./tempArray/L_true.npy")
print "L_true[pos1,pos2]: %s and L[pos1,pos2]: %s at position" % (L_true[pos1,pos2], L[pos1,pos2])

print "start analytical ..."
res = gradient.L_analyticalGradientII(gm, pos1,pos2, L_true, L, cqt_med, sigmas)
dJ12_anal = res.sum()
print "dJ12_anal: %s" % dJ12_anal

print "start numerical ..."
sigmas1 = sigmas.copy()
sigmas1[pos1,pos2] = sigmas1[pos1,pos2] + delta
sigmas1[pos2,pos1] = sigmas1[pos2,pos1] + delta
gm1 = RM.feature2GaussianMatrix(cqt_med, sigmas1) #(nSample, nFeature)
L1 = scipy.sparse.csgraph.laplacian(gm1, normed=True)
print "L1[pos1,pos1]: %s, L1[pos2,pos2]: %s" % (L1[pos1,pos1], L1[pos2,pos2])
J1 = 0.5 * (np.linalg.norm(L_true - L1))**2

sigmas2 = sigmas.copy()
sigmas2[pos1,pos2] = sigmas2[pos1,pos2] - delta
sigmas2[pos2,pos1] = sigmas2[pos2,pos1] - delta
gm2 = RM.feature2GaussianMatrix(cqt_med, sigmas2) #(nSample, nFeature)
L2 = scipy.sparse.csgraph.laplacian(gm2, normed=True)
print "L2[pos1,pos1]: %s, L2[pos2,pos2]: %s" % (L2[pos1,pos1], L2[pos2,pos2])
print "L_true[pos1,pos1]: %s, L_true[pos2,pos2]: %s" % (L_true[pos1,pos1], L_true[pos2,pos2])
J2 = 0.5 * (np.linalg.norm(L_true - L2))**2

dJ12_num = (J1-J2)/(2*delta)

print "start comparison ..."
print "dJ12_anal: %s" % dJ12_anal
print "dJ12_num: %s" % dJ12_num
print "diff percentage: %s" % ((dJ12_anal-dJ12_num)/dJ12_anal)


# print (gm > 0).all()
# for i, elm in enumerate(gm):
#   for j, elm2 in enumerate(elm):
#     if elm2 <= 0:
#       print "%s, %s" % (elm2, (i,j))
