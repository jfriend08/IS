import sys, scipy, time, os, librosa
import numpy as np
import scipy.io.wavfile as wav
from librosa.util import normalize

sys.path.append('../src')
import laplacian, gradient, plotGraph, librosaF
import RecurrenceMatrix as RM

np.random.seed(123)
delta = 1e-6

def testDLoverDW():
  w = np.random.rand(15, 15) + 1
  w = (w + w.T)/2
  L = scipy.sparse.csgraph.laplacian(w, normed=True)

  for i in xrange(10):
    for j in xrange(10):
      dL_num = gradient.L_analyticalGradient(w, i,j )

      wc1, wc2 = w.copy(), w.copy()
      wc1[i,j] += delta
      wc1[j,i] += delta
      wc2[i,j] -= delta
      wc2[j,i] -= delta
      dL_ana1 = scipy.sparse.csgraph.laplacian(wc1, normed=True)
      dL_ana2 = scipy.sparse.csgraph.laplacian(wc2, normed=True)
      dL_ana = (dL_ana1 - dL_ana2)/(2*delta)

      print "All small: %s" % ((dL_ana - dL_num) < 1e-10).all()

def testDWoverDSigma():
  nsample, nfeature = 100, 75
  sigma = np.random.rand(nsample, nsample) + 1
  sigma = (sigma + sigma.T)/2

  feature = np.random.rand(nsample, nfeature) + 2
  feature = normalize(feature, norm=2)
  for i in xrange(nsample):
    for j in xrange(nsample):
      dw_ana = gradient.dw_ij(i,j, sigma[i,j], feature)

      w1 = gradient.w_ij(i,j, sigma[i,j]+delta, feature)
      w2 = gradient.w_ij(i,j, sigma[i,j]-delta, feature)
      dw_num = (w1-w2)/(2*delta)

      print "is small: %s, dw_num: %s, dw_ana: %s" % (abs(dw_num-dw_ana) < 1e-10, dw_num, dw_ana)

def testWhereDLDsigmaDifferent():
  '''
  Looks like DJ/DSigma from numerical and analytical are different
  This is to test where does the difference occured
  '''
  def mp32np(fname):
    oname = 'temp.wav'
    cmd = 'lame --decode {0} {1}'.format( fname,oname )
    os.system(cmd)
    return wav.read(oname)
  sr, signal = mp32np('../data/audio/SALAMI_698.mp3')
  y = signal[:,0]

  if not os.path.exists("./tempArray/cqt_med.npy"):
    print "Perform beat_track and cqt"
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    cqt = librosa.cqt(y=y, sr=sr)
    cqt_med, frameConversion = librosaF.sync(cqt, beats, aggregate=np.median)
    cqt_med = cqt_med.T
    cqt_med = normalize(cqt_med, norm=2)
    np.save("./tempArray/cqt_med.npy", cqt_med)
  else:
    print "Loading cqt_med"
    cqt_med = np.load("./tempArray/cqt_med.npy")

  sigmas = np.random.rand(cqt_med.shape[0], cqt_med.shape[0]) + 1e-7
  sigmas = ((sigmas + sigmas.T)/2)

  gm = RM.feature2GaussianMatrix(cqt_med, sigmas) #(nSample, nFeature)
  L = scipy.sparse.csgraph.laplacian(gm, normed=True)
  L_true = np.load("./tempArray/L_true.npy")

  print "cqt_med [min, max]: %s" % str((cqt_med.min(), cqt_med.max()))
  print "sigmas [min, max]: %s" % str((sigmas.min(), sigmas.max()))
  print "gm [min, max]: %s" % str((gm.min(), gm.max()))
  print "L [min, max]: %s" % str((L.min(), L.max()))

  alpha = 2
  for i in xrange(gm.shape[0]):
    for j in xrange(i+1, gm.shape[0]):
      start_time = time.time()
      dJij_anal = gradient.L_analyticalGradientII(gm, i,j, L_true, L, cqt_med, sigmas)
      timeAnal = time.time() - start_time

      start_time = time.time()
      dJij_num = gradient.L_numericalGradientII(gm, i,j, L_true, L, cqt_med, sigmas, cqt_med)
      timeNum = time.time() - start_time

      start_time = time.time()
      dJij_anal_matrix = gradient.L_analyticalGradientII_getMatrix(gm, i,j, L_true, L, cqt_med, sigmas)
      dJij_anal_matrixSum = dJij_anal_matrix.sum()
      timeAnaMatrix = time.time() - start_time

      print "dJij_num: %s, dJij_anal: %s, dJij_anal_matrix.sum(): %s" % ( dJij_num, dJij_anal, dJij_anal_matrixSum)
      print "timeNum: %s, timeAnal:%s, timeAnaMatrix: %s\n" % (timeNum, timeAnal, timeAnaMatrix)

def testProperSigma():
  '''
  This is the test to find the proper sigma.
  Issue I got is the update for sigma is close to zero.
  Prob due to the issue of sigma or cqt normalization
  '''
  def mp32np(fname):
    oname = 'temp.wav'
    cmd = 'lame --decode {0} {1}'.format( fname,oname )
    os.system(cmd)
    return wav.read(oname)
  sr, signal = mp32np('../data/audio/SALAMI_698.mp3')
  y = signal[:,0]

  if not os.path.exists("./tempArray/cqt_med.npy"):
    print "Perform beat_track and cqt"
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    cqt = librosa.cqt(y=y, sr=sr)
    cqt_med, frameConversion = librosaF.sync(cqt, beats, aggregate=np.median)
    cqt_med = cqt_med.T
    cqt_med = normalize(cqt_med, norm=2)
    np.save("./tempArray/cqt_med.npy", cqt_med)
  else:
    print "Loading cqt_med"
    cqt_med = np.load("./tempArray/cqt_med.npy")

  sigmas = np.random.rand(cqt_med.shape[0], cqt_med.shape[0]) + 1e-7
  sigmas = ((sigmas + sigmas.T)/2)

  gm = RM.feature2GaussianMatrix(cqt_med, sigmas) #(nSample, nFeature)
  L = scipy.sparse.csgraph.laplacian(gm, normed=True)
  L_true = np.load("./tempArray/L_true.npy")

  print "cqt_med [min, max]: %s" % str((cqt_med.min(), cqt_med.max()))
  print "sigmas [min, max]: %s" % str((sigmas.min(), sigmas.max()))
  print "gm [min, max]: %s" % str((gm.min(), gm.max()))
  print "L [min, max]: %s" % str((L.min(), L.max()))

  alpha = 2
  result = {'isLijGettingCloser':[], 'isWholeLossSmaller':[]}
  for i in xrange(gm.shape[0]):
    for j in xrange(i+1, gm.shape[0]):
      if abs(L_true[i,j] - L[i,j]) < 1e-2:
        continue
      dJij_num = gradient.L_analyticalGradientII(gm, i,j, L_true, L, cqt_med, sigmas)
      dJij_anal = gradient.L_numericalGradientII(gm, i,j, L_true, L, cqt_med, sigmas, cqt_med)
      newSig = sigmas[i,j] - 1 * alpha * dJij_anal

      newgm = gm.copy()
      newgm[i,j] = gradient.w_ij(i,j,newSig,cqt_med)
      newgm[j,i] = newgm[i,j]
      newL = scipy.sparse.csgraph.laplacian(newgm, normed=True)

      isLijGettingCloser =  abs(L_true[i,j]-newL[i,j]) - abs(L_true[i,j]-L[i,j])
      isWholeLossSmaller = (0.5 * np.linalg.norm(L_true-newL)**2) - (0.5 * np.linalg.norm(L_true-L)**2)

      print "L_true[i,j]: %s, L[i,j]: %s, sigmas[i,j]: %s, Ana_update: %s, Num_update: %s" %( L_true[i,j], L[i,j], sigmas[i,j], -1 * alpha * dJij_anal, -1 * alpha * dJij_num)
      print "sigma from %s --> %s" % (sigmas[i,j], newSig)
      print "L_true[i,j]: %s, L[i,j] from %s --> %s, isLijGettingCloser: %s, isWholeLossSmaller: %s" % (L_true[i,j], L[i,j], newL[i,j], isLijGettingCloser, isWholeLossSmaller)
      print "update difference: %s percent\n" % ( ((-1 * alpha * dJij_num) - (-1 * alpha * dJij_anal))/(-1 * alpha * dJij_anal)  )

def testDLOSSoverDsigma():
  nsample, nfeature = 322, 75
  delta = 1e-8

  # cqt_med = np.load('./tempArray/cqt_med.npy')
  cqt_med = np.random.rand(nsample, nfeature) + 200
  cqt_med = normalize(cqt_med, norm=2)

  sigmas = np.random.rand(nsample, nsample)
  sigmas = ((sigmas + sigmas.T)/2)
  gm = RM.feature2GaussianMatrix(cqt_med, sigmas) #(nSample, nFeature)

  L = scipy.sparse.csgraph.laplacian(gm, normed=True)
  L_true = np.load("./tempArray/L_true.npy")

  for pos1 in xrange(nsample):
    for pos2 in xrange(nsample):
      if (L[pos1, pos2] - L_true[pos1, pos2]) < 1e-2:
        continue

      print "L_true[pos1,pos2]: %s and L[pos1,pos2]: %s at position" % (L_true[pos1,pos2], L[pos1,pos2])
      # print "start analytical ..."
      start_time = time.time()
      dJ12_anal = gradient.L_analyticalGradientII(gm, pos1,pos2, L_true, L, cqt_med, sigmas)
      # dJ12_anal = res.sum()
      print("--- Anal %s seconds ---" % (time.time() - start_time))
      print "dJ12_anal: %s" % dJ12_anal

      # print "start numerical ..."
      start_time = time.time()
      sigmas1 = sigmas.copy()
      sigmas1[pos1,pos2] = sigmas1[pos1,pos2] + delta
      sigmas1[pos2,pos1] = sigmas1[pos2,pos1] + delta
      gm1 = RM.feature2GaussianMatrix(cqt_med, sigmas1) #(nSample, nFeature)
      L1 = scipy.sparse.csgraph.laplacian(gm1, normed=True)
      # print "L1[pos1,pos1]: %s, L1[pos2,pos2]: %s" % (L1[pos1,pos1], L1[pos2,pos2])
      J1 = 0.5 * (np.linalg.norm(L_true - L1))**2

      sigmas2 = sigmas.copy()
      sigmas2[pos1,pos2] = sigmas2[pos1,pos2] - delta
      sigmas2[pos2,pos1] = sigmas2[pos2,pos1] - delta
      gm2 = RM.feature2GaussianMatrix(cqt_med, sigmas2) #(nSample, nFeature)
      L2 = scipy.sparse.csgraph.laplacian(gm2, normed=True)
      # print "L2[pos1,pos1]: %s, L2[pos2,pos2]: %s" % (L2[pos1,pos1], L2[pos2,pos2])
      # print "L_true[pos1,pos1]: %s, L_true[pos2,pos2]: %s" % (L_true[pos1,pos1], L_true[pos2,pos2])
      J2 = 0.5 * (np.linalg.norm(L_true - L2))**2
      print("--- Num %s seconds ---" % (time.time() - start_time))
      dJ12_num = (J1-J2)/(2*delta)

      # print "start comparison ..."
      print "dJ12_anal: %s, dJ12_num: %s, diff percentage: %s" % (dJ12_anal, dJ12_num, (dJ12_anal-dJ12_num)/dJ12_anal)

testWhereDLDsigmaDifferent()
