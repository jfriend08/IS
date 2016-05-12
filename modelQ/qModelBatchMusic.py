import sys, scipy, os, warnings, librosa, argparse, pickle
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from sklearn.feature_extraction import image
from librosa.util import normalize
import numpy as np
import scipy.io.wavfile as wav

sys.path.append('../src')
import laplacian, gradient, plotGraph, librosaF, qUpdate
import RecurrenceMatrix as RM

parser = argparse.ArgumentParser()
parser.add_argument("alpha", help="alpha for update step size")
parser.add_argument("namePrefix", help="name prefix for figures and files")
args = parser.parse_args()

'''All parameter should be just here'''
epco, res = 1000, []
np.random.seed(123)

qPath = "./Q/"
figurePath = "./fig/"

alpha = float(args.alpha)
namePrefix = args.namePrefix
namePrefix = namePrefix + "_Alpha" + str(alpha).replace(".", "_")
isBatch = True
analytical = False

qPath += namePrefix + '/'
figurePath += namePrefix + '/'

#check sigma and figure path, creat if not exist
if not os.path.exists(qPath):
  os.makedirs(qPath)
if not os.path.exists(figurePath):
  os.makedirs(figurePath)

print "Loading soundRecords ..."
soundRecords = pickle.load(open( "../data/soundRecords_workII.json", "rb" ))
print "soundRecords.keys()", soundRecords.keys()

print "Creating Q vector ..."
Q = np.random.rand(84) + 1e-7 #84 is cqt number of bins

for ep in xrange(epco):
  err = 0
  for sKey in soundRecords:
    cqt_med = soundRecords[sKey]["cqt_med"]
    gm = RM.featureQ2GaussianMatrix(cqt_med, Q) #(nSample, nFeature)
    L = scipy.sparse.csgraph.laplacian(gm, normed=True)
    m_true = soundRecords[sKey]["m_true"]
    L_true = soundRecords[sKey]["L_true"]

    err += 0.5 * np.linalg.norm(L_true-L)**2

    if isBatch:
      print "isBatch"
      Q = qUpdate.batchUpdate(gm, L, m_true, L_true, cqt_med, Q, alpha, figurePath, namePrefix, ep, analytical)
    else:
      print "isSingle"
      pass

  filename = qPath + namePrefix + "_step" + str(ep) + ".npy"
  print "saving Q to: ", filename
  np.save(filename, Q)

  err = float(err)/len(soundRecords) #average error amoung all sounds
  print "epoch: ", str(ep), " errors: ", str(err)
  res += [err]

  plotGraph.plotLine(figurePath + namePrefix + "_epch" + str(ep) + "_err", res, 'Error per epoch')
  plotGraph.plotLine(figurePath + namePrefix + "_epch" + str(ep) + "_Q", Q, 'Q per epoch', 2.5, -2)



