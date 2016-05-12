import sys, scipy, os, re, warnings, librosa, argparse, pickle
import librosa
import signal as Signal
import numpy as np
from librosa.util import normalize

sys.path.append('./')
import laplacian, gradient, plotGraph, librosaF, qUpdate
import RecurrenceMatrix as RM

SALAMI_AUDIO_DIR = "../data/SegmentationData/SALAMI/audio/"
SALAMI_ANNO_DIR = "../data/SegmentationData/SALAMI/data/"
JSON_FILE = "../data/soundRecordsII.json"
NUM_RECORDS = 100

TIME_LIMIT_PER_SONG = 600 #second, in case cqt runs for too long
SKIP_LIST = ["630", "640", "606", "674", "456"] #Black list of songs that will run cqt for too long
MAX_SIGNAL = 100000000 #limit for songs having long signal


soundRecords = {}

class TimeoutException(Exception):   # Custom exception class
    pass

def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException

# Change the behavior of SIGALRM
Signal.signal(Signal.SIGALRM, timeout_handler)

def runPreProp():
  for file in os.listdir(SALAMI_AUDIO_DIR):
    print '---------------------------'

    with open(JSON_FILE, 'wb') as handle:
      print "Saving soundRecords ..."
      pickle.dump(soundRecords, handle)

    if len(soundRecords) > NUM_RECORDS:
      return

    if file.endswith(".mp3"):

      sr, signal = librosaF.mp32np(SALAMI_AUDIO_DIR + file)
      print "loadCount: %s, file: %s, signal.shape: %s, sr: %s" % (len(soundRecords), file, signal.shape, sr)

      if signal.shape[0] > MAX_SIGNAL:
        print "Signal too big. Skipping ..."
        continue

      #for song has signal
      if signal.shape[1] != 0:
        soundID = re.split('_|\.',file)[0]

        print "==> soundID: %s, isInList: %s " % (soundID, soundID in SKIP_LIST)
        if soundID in SKIP_LIST:
          print "file is in SKIP_LIST. Skipping ..."
          continue

        y = signal[:,0]

        Signal.alarm(TIME_LIMIT_PER_SONG)
        try:
          print "beat track ..."
          tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
          print "cqt ..."
          cqt = librosa.cqt(y=y, sr=sr)
        except TimeoutException:
          print "TimeoutException after 30 second"
          continue
        else:
          print "reset alarm"
          Signal.alarm(0)

        print "sync ..."
        cqt_med, frameConversion = librosaF.sync(cqt, beats, aggregate=np.median)
        cqt_med = cqt_med.T
        cqt_med = normalize(cqt_med, norm=2)

        print "Interval2Frame ..."
        interval = librosaF.loadInterval2Frame(SALAMI_ANNO_DIR+soundID+'/parsed/textfile1_uppercase.txt', sr, frameConversion)

        m_true = RM.label2RecurrenceMatrix("../data/2.jams", cqt_med.shape[0], interval)
        L_true = scipy.sparse.csgraph.laplacian(m_true, normed=True)

        print "file: %s, beats.shape: %s, m_true.shape: %s" % (file, beats.shape, m_true.shape)
        soundRecords[soundID] = {'cqt_med': None, 'm_true': None, 'L_true': None}
        soundRecords[soundID]['cqt_med'] = cqt_med
        soundRecords[soundID]['m_true'] = m_true
        soundRecords[soundID]['L_true'] = L_true

runPreProp()