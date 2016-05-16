import sys, scipy, os, warnings, librosa, argparse, re
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
parser.add_argument("Q_dir", help="directory for each of Q")
parser.add_argument("out_dir", help="output directory")
args = parser.parse_args()
Q_dir = args.Q_dir
out_dir = args.out_dir

eachQ, count = {}, 0
minV, maxV = 0, 0

for file in os.listdir(Q_dir):
  count += 1
  name = re.split('step|\.', file)[1]
  print "name: ", name
  eachQ[name] = np.load(Q_dir+file)
  minV = min(minV, eachQ[name].min())
  maxV = max(maxV, eachQ[name].max())

for ep in xrange(1, count):
  print "plot ep: %s" % (ep)
  plotGraph.plotLine(out_dir + "epch" + str(ep) + "_Q", eachQ[str(ep)], 'Q per epoch', maxV, minV)


