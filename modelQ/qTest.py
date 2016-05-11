import sys, scipy, os, warnings, librosa, argparse
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from sklearn.feature_extraction import image
from librosa.util import normalize
import numpy as np
import scipy.io.wavfile as wav

sys.path.append('../src')
import laplacian, gradient, plotGraph, librosaF, sigmaUpdate
import RecurrenceMatrix as RM


cqt_med = np.load("../model/tempArray/cqt_med.npy")
q = np.random.rand(cqt_med.shape[1])
# cqt_med = np.array([[i]*5 for i in xrange(5)])
# q = np.array([2]*5)
gm = RM.featureQ2GaussianMatrix(cqt_med, q)
L = scipy.sparse.csgraph.laplacian(gm, normed=True)
print L