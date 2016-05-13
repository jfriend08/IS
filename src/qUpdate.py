import numpy as np
import sys, scipy, os, warnings, librosa, argparse
sys.path.append('../src')
import laplacian, gradient, plotGraph, librosaF
import RecurrenceMatrix as RM

def batchUpdate(gm, L, m_true, L_true, cqt_med, Q, alpha, figurePath, namePrefix, epoch, analytical=True):
  Q_copy = Q.copy()

  for qidx, q in enumerate(Q_copy):
    if analytical:
      dJ_dq = gradient.L_analyticalGradientQII(L_true, L, gm, Q, qidx, cqt_med)
    else:
      dJ_dq = gradient.L_numericalGradientQII(L_true, Q, qidx, cqt_med) #use original Q to update each one

    Q_copy[qidx] = Q_copy[qidx] - alpha * dJ_dq
    # print "qidx: %s, orig Q: %s, new Q: %s, dJ_dq: %s" % (qidx, Q[qidx], Q_copy[qidx], dJ_dq)

  return Q_copy