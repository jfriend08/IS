import numpy as np

def updateSomething(m):
  m[1,2] = 4
  m[2,1] = 4
  return m


def main():
  sigmas = np.random.rand(5, 5) + 1e-7 #add a base in case of 0 sigma
  sigmas = (sigmas + sigmas.T)/2
  updateSomething(sigmas)
  print sigmas

main()
