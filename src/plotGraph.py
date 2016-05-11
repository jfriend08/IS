import numpy as np
import matplotlib.pyplot as plt

def plot1(path, obj1, name1):
  plt.figure(figsize=(7.5, 5))
  plt.subplot(1, 1, 1)
  plt.pcolor(obj1, cmap="viridis")
  plt.colorbar()
  plt.title(name1)
  plt.savefig(path)
  plt.close()

def plot2(path, obj1, name1, obj2, name2):
  plt.figure(figsize=(15, 5))
  plt.subplot(1, 2, 1)
  plt.pcolor(obj1, cmap="viridis")
  plt.colorbar()
  plt.title(name1)

  plt.subplot(1, 2, 2)
  plt.pcolor(obj2, cmap="viridis")
  plt.colorbar()
  plt.title(name2)
  plt.savefig(path)
  plt.close()

def plot4(path, obj1, name1, obj2, name2, obj3, name3, obj4, name4):
  plt.figure(figsize=(15, 10))
  plt.subplot(2, 2, 1)
  plt.pcolor(obj1, cmap="viridis")
  plt.colorbar()
  plt.title(name1)

  plt.subplot(2, 2, 2)
  plt.pcolor(obj2, cmap="viridis")
  plt.colorbar()
  plt.title(name2)
  plt.savefig(path)

  plt.subplot(2, 2, 3)
  plt.pcolor(obj3, cmap="viridis")
  plt.colorbar()
  plt.title(name3)

  plt.subplot(2, 2, 4)
  plt.pcolor(obj4, cmap="viridis")
  plt.colorbar()
  plt.title(name4)

  plt.savefig(path)
  plt.close()

def plotLine(path, obj1, name1):
  plt.figure(figsize=(15,5))
  plt.plot(range(len(obj1)), obj1, '--')
  plt.title(name1)

  plt.savefig(path)
  plt.close()