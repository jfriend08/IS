import numpy as np
import sys, scipy

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.feature_extraction import image
import librosa

sys.path.append('../src')
import laplacian
import RecurrenceMatrix as RM
import gradient, plotGraph

mu, sigma, length = 10, 1, 22050/2 # mean, standard deviation, and sampling rate
epco, alpha, res = 30, 2000, []
np.random.seed(123)

sigmaPath = "./sigmas/"
figurePath = "./fig/"
namePrefix = "modelRun_Alpha" + str(alpha)

signal = []
for i in xrange(20):
  if i%2==0:
    signal += list(np.random.normal(mu, sigma, length))
  else:
    signal += list(np.random.normal(-mu, sigma, length))


signal = np.array(signal)
plt.figure(figsize=(15, 5))
plt.title('Signal Wave...')
plt.plot(signal)
plt.ylim((-15,15))
plt.savefig('./fig/signal.png')

############################################################
cqt = np.transpose(librosa.cqt(signal))
print signal.shape, cqt.shape


sigmas = np.random.rand(cqt.shape[0], cqt.shape[0])
sigmas = (sigmas + sigmas.T)/2
gm = RM.feature2GaussianMatrix(cqt, sigmas)
print gm.shape
print "is symmetric: ", (gm==np.transpose(gm)).all()

############################################################
term = ['up', 'down']
term2 = ['transitDown', 'transitUp']
transitDelta = 50
allInterval = []

#Create interval base on the signal we created
for i in xrange(20):
  allInterval += [[np.array([length*i+transitDelta, length*(i+1)-transitDelta]).astype(float), term[i%2]]]
  allInterval += [[np.array([length*(i+1)-transitDelta, length*(i+1)+transitDelta]).astype(float), term2[i%2]]]
allInterval.pop()

#Convert interval from time to frame
def t2f(x):
    return librosa.core.time_to_frames(x, sr=22050, hop_length=512, n_fft=None)
allInterval = [[np.apply_along_axis(t2f, 0, elm[0]/22050), elm[1]] for elm in allInterval]
print allInterval

m_true = RM.label2RecurrenceMatrix("../data/2.jams", gm.shape[0], allInterval)
print "m_true is symmetruc: ", (m_true==m_true.T).all()
############################################################
L = scipy.sparse.csgraph.laplacian(gm, normed=True)
L_true = scipy.sparse.csgraph.laplacian(m_true, normed=True)

print "L diag all 1: ", all([L[i,i] == 1 for i in xrange(L.shape[0])])
print "L_true diag all 1: ", all([L_true[i,i] == 1 for i in xrange(L_true.shape[0])])

err = 0.5 * np.linalg.norm(L_true-L)**2

res += [err]

filename = figurePath + namePrefix + "_original.png"
plotGraph.plot2(filename, m_true, "m_true", gm, "gm")

for i in xrange(epco):
  accu = gradient.allDLoss(sigmas, L, L_true, gm, cqt)
  sigmas = sigmas - alpha * accu
  sigmas = np.around(sigmas, decimals = 10)

  filename = sigmaPath + namePrefix + "_step" + str(i) + ".npy"
  print "saving sigmas to: ", filename
  np.save(filename, sigmas)

  gm = RM.feature2GaussianMatrix(cqt, sigmas)
  L = scipy.sparse.csgraph.laplacian(gm, normed=True)

  filename = figurePath + namePrefix + "_epch" + str(i) + ".png"
  plotGraph.plot2(filename, m_true, "m_true", gm, "gm")

  err = 0.5 * np.linalg.norm(L_true-L)**2
  res += [err]
print res




