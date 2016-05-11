import numpy as np

def getCircleMatrix():
  #make circle in 2D matrix
  l = 100
  x, y = np.indices((l, l))

  center1 = (28, 28)
  center2 = (40, 50)
  center3 = (58, 58)
  center4 = (24, 70)

  radius1, radius2, radius3, radius4 = 16, 14, 5, 14

  circle1 = (x - center1[0]) ** 2 + (y - center1[1]) ** 2 < radius1 ** 2
  circle2 = (x - center2[0]) ** 2 + (y - center2[1]) ** 2 < radius2 ** 2
  circle3 = (x - center3[0]) ** 2 + (y - center3[1]) ** 2 < radius3 ** 2
  circle4 = (x - center4[0]) ** 2 + (y - center4[1]) ** 2 < radius4 ** 2
  #add value
  img = circle1 + circle3
  mask = img.astype(bool)
  img = img.astype(float)

  # img += 1 + 0.1 * np.random.randn(*img.shape)
  return img

def getDiagMatrix():
  a = np.zeros((100, 100), int)
  np.fill_diagonal(a, 5)
  return a

def getOffDiagMatrix(thick=1):
  a = np.zeros((100, 100), int)
  np.fill_diagonal(a, 5)
  for i in [1, 10, 20, 30, 40, 50, 60, 70, 80]:
    for j in xrange(i-thick, i+thick):
      if 0 <= i < 100 and 0 <= j < 100:
        a[i][j] = 5
  return a

def getOffDiagMatrixII(thick=1):
  a = np.ones((1, 100))[0]
  m = np.diag(a, 0)
  for t in xrange(1, thick+1):
    b = np.ones((1, 100-t))[0]
    m += np.diag(b, -t) + np.diag(b, t)
  return m

#Similar to getOffDiagMatrixIV, different cutting. is symmetric
def getOffDiagMatrixIII(thick=1, nBreak=0):
  a = np.ones((1, 100))[0]
  m = np.diag(a, 0)

  leng = len(m[0])/nBreak
  for t in xrange(1, thick+1):
    idx = 0
    b = np.ones((1, 100-t))[0]
    while idx*leng < len(m[0]):
      try:
        b[idx*leng-3:idx*leng+3-t] = 0
      except:
        break
      idx += 1
    m += np.diag(b, -t) + np.diag(b, t)
  return m

#Similar to getOffDiagMatrixIII, but different cutting. not symmetric
def getOffDiagMatrixIV(thick=1, nBreak=0):
  a = np.ones((1, 100))[0]
  m = np.diag(a, 0)
  for t in xrange(1, thick+1):
    b = np.ones((1, 100-t))[0]
    m += np.diag(b, -t) + np.diag(b, t)

  mybreak, idx, leng = np.zeros((len(m[0]),6)), 1, len(m[0])/nBreak
  while idx*leng < len(m[0]):
    m[:,idx*leng-3:idx*leng+3] = mybreak
    idx += 1
  return m

