import numpy as np
import sys, scipy, os, warnings, librosa, argparse
sys.path.append('../src')
import laplacian, gradient, plotGraph, librosaF
import RecurrenceMatrix as RM

def updateTargetEachOne(gm, L, L_true, cqt_med, sigmas, res, alpha, figurePath, namePrefix):
  '''
  ToDo: need to check its correctness
  '''
  updateC, loop = 0, 1000
  while loop > 0:
    loop -= 1

    print 'where the max diff in L ...'
    diffL = np.absolute(L_true-L)
    x,y = np.unravel_index(diffL.argmax(), diffL.shape)
    print "diffL: %s, L_true: %s, L: %s" % ((diffL[x,y]), L_true[x,y], L[x,y])
    print (x,y)

    dL = gradient.L_analyticalGradient(gm,x,y)
    dw = gradient.dw_ij(x,y,sigmas[x,y],cqt_med)
    print "dL(x,y): %s" %dL[x,y]
    print "dw: %s" % dw
    print 'cqt_med l2 diff is: %s' % np.linalg.norm(cqt_med[x]-cqt_med[y])
    print 'w[x,y] is: %s, sigmas[x,y] is: %s' % (gm[x,y], sigmas[x,y])

    diff = alpha * ( -1 * 2 * (L_true - L) * dL  * dw )

    print 'diff-alpha will be ...'
    maxdiff = np.absolute(diff)
    print "the alpha-diff at location is: %s" % maxdiff[x,y]
    x,y = np.unravel_index(maxdiff.argmax(), maxdiff.shape)
    print "max in diff is: %s, location is: %s" % (np.amax(maxdiff), (x,y))

    sigmas = sigmas - diff
    gm = RM.feature2GaussianMatrix(cqt_med, sigmas)
    L = scipy.sparse.csgraph.laplacian(gm, normed=True)

    print 'after update. L diff at location: %s is %s' % (str((x,y)), L_true[x,y] - L[x,y])
    print '\n'
  return sigmas

def updateBatch(gm, L, m_true, L_true, cqt_med, sigmas, alpha, figurePath, namePrefix, epoch, analytical=True):
  res = sigmas.copy()
  for i in xrange(gm.shape[0]):
    for j in xrange(i+1, gm.shape[0]):
      if abs(L_true[i,j] - L[i,j]) < 1e-3: #only update the sigma where L-difference is large
        continue

      if analytical:
        # print "analytical"
        dJij_dSigma = gradient.L_analyticalGradientII_getMatrix(gm, i,j, L_true, L, cqt_med, sigmas)
      else:
        # print "numerical"
        dJij_dSigma = gradient.L_numericalGradientII(gm, i,j, L_true, L, cqt_med, sigmas, cqt_med)

      res[i,j] = sigmas[i,j] - alpha * dJij_dSigma
      res[j,i] = sigmas[j,i] - alpha * dJij_dSigma

  return res

def updateEachOne(gm, L, m_true, L_true, cqt_med, sigmas, alpha, figurePath, namePrefix, epoch, analytical=True):
  res, updateC = [], 0
  sigmasCopy = sigmas.copy()
  for i in xrange(gm.shape[0]):
    for j in xrange(i+1, gm.shape[0]):
      if abs(L_true[i,j] - L[i,j]) > 1e-2: #only update the sigma where L-difference is large
        updateC += 1

        if analytical:
          # print "analytical"
          analMatix = gradient.L_analyticalGradientII_getMatrix(gm, i,j, L_true, L, cqt_med, sigmasCopy)
          dJij_dSigma = analMatix.sum()
        else:
          # print "numerical"
          dJij_dSigma = gradient.L_numericalGradientII(gm, i,j, L_true, L, cqt_med, sigmasCopy, cqt_med)

        sigmasCopy[i,j] = sigmasCopy[i,j] - alpha * dJij_dSigma
        sigmasCopy[j,i] = sigmasCopy[j,i] - alpha * dJij_dSigma

        gm = RM.feature2GaussianMatrix(cqt_med, sigmasCopy)
        L = scipy.sparse.csgraph.laplacian(gm, normed=True)

        err = 0.5 * np.linalg.norm(L_true-L)**2
        res += [err]
        print "cur err: %s at: %s \n" % (str(err), str((i,j)))

        if updateC % 1000 == 1:
          filename = figurePath + namePrefix + "_epoch" + str(epoch) + "_update" + str((i,j))
          plotGraph.plot1(filename + "Sigma", sigmasCopy, "sigmas_" + str((i,j)))
          plotGraph.plot4(filename, m_true, "m_true", gm, "gm", L_true, "L_true", L, "L")
          plotGraph.plotLine(figurePath + namePrefix + "_err", res, 'Error per Steps -- Validation')
  return sigmasCopy
