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


parser = argparse.ArgumentParser()
parser.add_argument("alpha", help="alpha for update step size")
parser.add_argument("namePrefix", help="name prefix for figures and files")
args = parser.parse_args()


'''All parameter should be just here'''
epco, res = 50, []
np.random.seed(123)

sigmaPath = "./sigmas/"
figurePath = "./fig/"

alpha = int(args.alpha)
namePrefix = args.namePrefix
namePrefix = namePrefix + "_Alpha" + str(alpha)
isBatch = False
analytical = True

sigmaPath += namePrefix + '/'
figurePath += namePrefix + '/'


#check sigma and figure path, creat if not exist
if not os.path.exists(sigmaPath):
  os.makedirs(sigmaPath)
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

print "Creating sigmas matrix ..."
sigmas = np.random.rand(cqt_med.shape[0], cqt_med.shape[0]) + 1e-7 #add a base in case of 0 sigma
sigmas = ((sigmas + sigmas.T)/2)

gm = RM.feature2GaussianMatrix(cqt_med, sigmas) #(nSample, nFeature)
L = scipy.sparse.csgraph.laplacian(gm, normed=True)
m_true = RM.label2RecurrenceMatrix("../data/2.jams", gm.shape[0], interval)
L_true = scipy.sparse.csgraph.laplacian(m_true, normed=True)
np.save("./tempArray/L_true.npy", L_true)

print "cqt_med [min, max]: %s" % str((cqt_med.min(), cqt_med.max()))
print "sigmas [min, max]: %s" % str((sigmas.min(), sigmas.max()))
print "gm [min, max]: %s" % str((gm.min(), gm.max()))
print "L [min, max]: %s" % str((L.min(), L.max()))

filename = figurePath + namePrefix + "_orig.png"
plotGraph.plot4(filename, m_true, "m_true", gm, "gm", L_true, "L_true", L, "L")

err = 0.5 * np.linalg.norm(L_true-L)**2
res += [err]
print "errors: ", str(err)

for ep in xrange(epco):
  '''update sigma'''
  if isBatch:
    print "isBatch"
    sigmas = sigmaUpdate.updateBatch(gm, L, m_true, L_true, cqt_med, sigmas, alpha, figurePath, namePrefix, ep, analytical)
  else:
    print "single update"
    sigmas = sigmaUpdate.updateEachOne(gm, L, m_true, L_true, cqt_med, sigmas, alpha, figurePath, namePrefix, ep, analytical)
    # sigmas = updateTargetEachOne(gm, L, L_true, cqt_med, sigmas, res)

  gm = RM.feature2GaussianMatrix(cqt_med, sigmas)
  L = scipy.sparse.csgraph.laplacian(gm, normed=True)

  filename = sigmaPath + namePrefix + "_step" + str(ep) + ".npy"
  print "saving sigmas to: ", filename
  np.save(filename, sigmas)

  filename = figurePath + "/" + namePrefix + "_epch" + str(ep)
  plotGraph.plot4(filename, m_true, "m_true", gm, "gm", L_true, "L_true", L, "L")

  err = 0.5 * np.linalg.norm(L_true-L)**2
  print "epoch: ", str(ep), " errors: ", str(err)
  res += [err]
  plotGraph.plotLine(figurePath + namePrefix + "_epch" + str(ep) + "_err", res, 'Error per epoch')

print res








# # print "Perform subsegment"
# # sub_beats = librosa.segment.subsegment(cqt, beats)
# # cqt_med_sub = librosa.util.sync(cqt, sub_beats, aggregate=np.median)


# # plt.figure()
# # plt.subplot(3, 1, 1)
# # librosa.display.specshow(librosa.logamplitude(cqt**2,ref_power=np.max),x_axis='time',sr=sr)
# # plt.colorbar(format='%+2.0f dB')
# # plt.title('CQT power, shape={}'.format(cqt.shape))

# # plt.subplot(3, 1, 2)
# # librosa.display.specshow(librosa.logamplitude(cqt_med**2,ref_power=np.max))
# # plt.colorbar(format='%+2.0f dB')
# # plt.title('Beat synchronous CQT power, ,shape={}'.format(cqt_med.shape))

# # plt.subplot(3, 1, 3)
# # librosa.display.specshow(librosa.logamplitude(cqt_med_sub**2,ref_power=np.max))

# # plt.colorbar(format='%+2.0f dB')
# # plt.title('Sub-beat synchronous CQT power, shape={}'.format(cqt_med_sub.shape))
# # plt.tight_layout()
# # plt.savefig('./test.png')