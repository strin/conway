#names = ['series', 'series2', 'series3', 'series4']
#colors = ['r', 'g', 'b', 'm']
import sys
import os
import numpy as np
colors = ['r', 
          'b',
          'g',
          'm',
          'k',
         ]
style = ['x-', '.-']
name = sys.argv[1]
num_series = int(sys.argv[2])
num_plots = int(sys.argv[3])
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.hold(True)

def readlines(filename):
  f = open(filename, 'r')
  nums = []
  for line in f:
    nums.append(float(line.rstrip(',\n')))
  f.close()
  return nums

def readmeta(filename, key):
  return str(os.popen("awk '/%s/ {print $2}' %s" % (key, filename)).read().replace('\n', ''))

for n in range(num_series-num_plots, num_series):
  ni = n-num_series+num_plots
  testerr = readlines('result/%s/test-%d' % (name, n))
  trainerr = readlines('result/%s/train-%d' % (name, n))
  logfile = 'result/%s/log-%d' % (name, n)
  meta = list()
  keys = ['T', 'Q', 'B', 'num_hidden']
  for key in keys:
    meta.append(readmeta(logfile, key))
  testlag = float(readmeta(logfile, 'testlag'))
  train_size = float(readmeta(logfile, 'num_train_data'))

  if len(trainerr) > 0:
    plt.figure(1)
    plt.plot(np.array(range(len(trainerr)))*float(testlag)/float(train_size), trainerr, '%s%s' % (colors[ni % len(colors)], style[ni / len(colors)]), label=' '.join(meta))
  if len(testerr) > 0:
    plt.figure(2)
    plt.plot(np.array(range(len(testerr)))*float(testlag)/float(train_size), testerr, '%s%s' % (colors[ni % len(colors)], style[ni / len(colors)]), label=' '.join(meta))

if len(trainerr) > 0:
  plt.figure(1)
  plt.title(name+" (training error) ")
  plt.xlabel('Effective passes through training data')
  plt.ylabel('Accuracy')
  plt.axis()
  plt.legend(loc=4,prop={'size':9})
  plt.savefig('plot_train.pdf')

if len(testerr) > 0:
  plt.figure(2)
  plt.title(name+" (test error) ")
  plt.xlabel('Effective passes through training data')
  plt.ylabel('Accuracy')
  plt.axis()
  plt.legend(loc=4,prop={'size':9})
  plt.savefig('plot.pdf')

