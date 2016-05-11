import sys, scipy, os, warnings, librosa, argparse
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

if not os.path.exists("./tempArray/cqt.npy"):
  sr, signal = librosaF.mp32np('../data/audio/SALAMI_698.mp3')
  y = signal[:,0]
  print "Perform beat_track and cqt"
  tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
  cqt = librosa.cqt(y=y, sr=sr)
  print "saving cqt and beats... "
  np.save("./tempArray/cqt.npy", cqt)
  np.save("./tempArray/beats.npy", beats)
else:
  print "Loading cqt_med and frameConversion... "
  cqt = np.load('./tempArray/cqt.npy')
  beats = np.load('./tempArray/beats.npy')
  sr = 44100

print "Perform sync ..."
cqt_med, frameConversion = librosaF.sync(cqt, beats, aggregate=np.median)
cqt_med = cqt_med.T
cqt_med = normalize(cqt_med, norm=2)

print "Perform loadInterval2Frame ..."
interval = librosaF.loadInterval2Frame("../data/anno/698/parsed/textfile1_uppercase.txt", sr, frameConversion)

print "Creating Q vector ..."
Q = np.random.rand(cqt_med.shape[1]) + 1e-7

gm = RM.featureQ2GaussianMatrix(cqt_med, Q) #(nSample, nFeature)
L = scipy.sparse.csgraph.laplacian(gm, normed=True)
m_true = RM.label2RecurrenceMatrix("../data/2.jams", gm.shape[0], interval)
L_true = scipy.sparse.csgraph.laplacian(m_true, normed=True)
np.save("./tempArray/L_true.npy", L_true)

print "cqt_med [min, max]: %s" % str((cqt_med.min(), cqt_med.max()))
print "Q [min, max]: %s" % str((Q.min(), Q.max()))
print "gm [min, max]: %s" % str((gm.min(), gm.max()))
print "L [min, max]: %s" % str((L.min(), L.max()))

filename = figurePath + namePrefix + "_orig.png"
plotGraph.plot4(filename, m_true, "m_true", gm, "gm", L_true, "L_true", L, "L")

err = 0.5 * np.linalg.norm(L_true-L)**2
res += [err]
print "errors: ", str(err)

for ep in xrange(epco):
  '''update Q'''
  if isBatch:
    print "isBatch"
    Q = qUpdate.batchUpdate(gm, L, m_true, L_true, cqt_med, Q, alpha, figurePath, namePrefix, ep, analytical)
  else:
    print "isSingle"
    pass

  gm = RM.featureQ2GaussianMatrix(cqt_med, Q)
  L = scipy.sparse.csgraph.laplacian(gm, normed=True)

  filename = qPath + namePrefix + "_step" + str(ep) + ".npy"
  print "saving Q to: ", filename
  np.save(filename, Q)

  filename = figurePath + "/" + namePrefix + "_epch" + str(ep)
  plotGraph.plot4(filename, m_true, "m_true", gm, "gm", L_true, "L_true", L, "L")

  err = 0.5 * np.linalg.norm(L_true-L)**2
  print "epoch: ", str(ep), " errors: ", str(err)
  res += [err]
  plotGraph.plotLine(figurePath + namePrefix + "_epch" + str(ep) + "_err", res, 'Error per epoch')
  plotGraph.plotLine(figurePath + namePrefix + "_epch" + str(ep) + "_Q", Q, 'Q per epoch')


