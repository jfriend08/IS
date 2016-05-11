import subprocess
import numpy as np


job = 's48'
time = "24:00:00"
ram = "8GB"
l_cmd = "walltime=%s,mem=%s"%(time, ram)
for i in xrange(1, 21, 1):
  alpha = i * 0.5
  jobName = 'qTest_num'
  subprocess.call('qsub runModel.pbs -N {0} -q {1} -l {2} -v alpha={3},namePrefix={4}'.format(
          jobName, job, l_cmd, alpha, jobName), shell=True)