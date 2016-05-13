import sys, scipy, os, warnings, librosa, argparse, pickle, time
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from sklearn.feature_extraction import image
from librosa.util import normalize
import numpy as np
import scipy.io.wavfile as wav

sys.path.append('../src')
import laplacian, gradient, plotGraph, librosaF, qUpdate
import RecurrenceMatrix as RM

np.random.seed(123)

def testNumericalAnalyticalGradient():
  print "Loading soundRecords ..."
  soundRecords = pickle.load(open( "../data/soundRecords_workII.json", "rb" ))
  print "soundRecords.keys()", soundRecords.keys()

  print "Creating Q vector ..."
  Q = np.random.rand(84) + 1e-7 #84 is cqt number of bins

  print "Select song ..."
  sKey = soundRecords.keys()[1]
  cqt_med = soundRecords[sKey]["cqt_med"]
  gm = RM.featureQ2GaussianMatrix(cqt_med, Q) #(nSample, nFeature)
  L = scipy.sparse.csgraph.laplacian(gm, normed=True)
  m_true = soundRecords[sKey]["m_true"]
  L_true = soundRecords[sKey]["L_true"]
  print "cqt_med.shape: %s, gm.shape: %s, L.shape: %s, m_true.shape: %s, L_true.shape: %s" % (cqt_med.shape, gm.shape, L.shape, m_true.shape, L_true.shape)
  # plotGraph.plot4("./test.png", m_true, "m_true", gm, "gm", L_true, "L_true", L, "L")
  res = []
  for qidx in xrange(Q.shape[0]):
    start_time = time.time()
    dJ_dq_num = gradient.L_numericalGradientQII(L_true, Q, qidx, cqt_med) #use original Q to update each one
    time_num = time.time() - start_time

    start_time = time.time()
    dJ_dq_ana = gradient.L_analyticalGradientQII(L_true, L, gm, Q, qidx, cqt_med)
    time_ana = time.time() - start_time
    err = abs(dJ_dq_ana - dJ_dq_num)/max(abs(dJ_dq_ana), abs(dJ_dq_num))
    res += [err]

    print "qidx: %s, dJ_dq_ana: %s, dJ_dq_num: %s, relative error: %s" % (qidx, dJ_dq_ana, dJ_dq_num, err)
    print "time_num: %s, time_ana: %s\n" % (time_num, time_ana)

  plotGraph.plotLine("Ana_vs_num_relativeErr_Q", res, 'Error per try', 1e-5, 0)

testNumericalAnalyticalGradient()