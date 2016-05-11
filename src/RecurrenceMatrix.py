import numpy as np
import math, sys
import librosa
import jams

sys.path.append('../src')
import gradient

def construct(labels):
  '''
  Construct recurrence matrix from kmeans labels
  '''
  a = np.ones((1, labels.shape[0]))[0]
  R = np.diag(a, 0)
  for i in xrange(len(labels)-1):
    label = labels[i]
    for j in [j for j in xrange(i+1, len(labels)) if labels[i] == labels[j]]:
        R[i][j] = 1
        R[j][i] = 1
  return R

def adjacentMatrix(size):
  '''
  Building adjacent matrix as the paper mentioned
  '''
  res = np.array([ [0.0]*size for _ in xrange(size)])
  for i in xrange(size):
    if i-1 >=0:
      res[i][i-1] = 1
    if i+1 < size:
      res[i][i+1] = 1
  return res

def sumEachRow(m):
  r, c = m.shape
  res = np.zeros(r)
  for i in xrange(r):
    res[i] = m[i,:]
  return res

def getBalancedMiu(R, delta):
  '''
  Calculate the miu to balance the linkage of recurrence and adjacent matrix
  '''
  d_R = np.apply_along_axis(sum, 1, R)
  d_delta = np.apply_along_axis(sum, 1, delta)
  return np.dot(d_R,d_R+d_delta)/np.power(np.linalg.norm(d_R + d_delta), 2)

def gaussianKernel(m, c=0.5):
  numSample, numFeature = m.shape
  res = np.array([ [0.0]*numSample for _ in xrange(numSample)])
  for i in xrange(numSample):
    for j in xrange(i, numSample):
      diff = np.power(np.linalg.norm(m[i,:] - m[j,:]), 2)
      val = np.exp((-1*diff/(2*c**2)))
      res[i][j] = val
      res[j][i] = val
  return res

def featureQ2GaussianMatrix(feature, Q):
  if not (feature.shape[1]==Q.shape[0]):
    raise ValueError('Q not match with feature')

  nSample, nFeature = feature.shape
  a = np.ones((1, nSample))[0]
  m = np.diag(a, 0)
  for i in xrange(nSample):
    for j in xrange(i+1, nSample):
      diff = feature[i,:] - feature[j,:]
      val = (diff*diff*Q).sum()
      # print "diff: %s, val: %s" % (diff, val)
      m[i,j] = val
      m[j,i] = val
  return m

def feature2GaussianMatrix(feature, sigmas):
  '''
  Given features along time frame, and returnning the pair-wise gaussian similarity
  @para {feature}: feature matrix in shape of (nSample, nFeature)
  @para {sigmas}: learnable parameter to calculate gaussian similarity
  @para {return}: recurrence similarity matrix
  '''
  if not (sigmas==np.transpose(sigmas)).all():
    raise ValueError('input sigmas matrix not symmetirc')

  nSample, nFeature = feature.shape
  minDiff, maxDiff = sys.float_info.max, sys.float_info.min
  a = np.ones((1, nSample))[0]
  m = np.diag(a, 0)
  for i in xrange(nSample):
    for j in xrange(i+1, nSample):
      diff = np.linalg.norm(feature[i]-feature[j])
      minDiff, maxDiff = min(minDiff, diff), max(maxDiff, diff)
      val = gradient.w_ij(i, j, sigmas[i,j], feature) #call value from gradient module
      m[i,j] = val
      m[j,i] = val
  return m

def getIntervalFromJAMS(path):
  j = jams.load(path)
  res = []
  for i in zip(list(j.annotations[0].data.time), list(j.annotations[0].data.time + j.annotations[0].data.duration), j.annotations[0].data.value):
    v = [[librosa.time_to_frames([i[0].total_seconds(), i[1].total_seconds()]), i[2].encode("ascii")]]
    res += v
  return res

def label2RecurrenceMatrix(jamsPath, matrixSize, intervals=[]):
  if len(intervals) == 0:
    intervals = getIntervalFromJAMS(jamsPath)
  a = np.ones((1, matrixSize))[0]
  m = np.diag(a, 0)
  allterms = {} #collecting all terms and its corresponding intervals
  for i, term in intervals:
    if term in allterms:
      allterms[term] += range(i[0], i[1]+1)
    else:
      allterms[term] = range(i[0], i[1]+1)
  for k in allterms: #marking 1 for all pair of time points in interval
    times = allterms[k]
    for i in xrange(len(times)):
      for j in xrange(i+1, len(times)):
        t1, t2 = times[i], times[j]
        m[t1,t2] = 1
        m[t2,t1] = 1
  return m



# allLoss(1, 1, 1)

# # m = np.zeros((40, 84))
# m = np.matrix([[1,2,3],[2,2,4], [1,2,3], [2,2,4], [1,2,3]])
# print feature2GaussianMatrix(m, 1)
# getIntervalFromJAMS("../data/2.jams")


