import csv
import numpy.random as npr
import numpy as np
# import matplotlib.pyplot as plt
import os, sys
from optparse import OptionParser

toInt = np.vectorize(int)
H = 20
W = 20

def sigmoid(x):
  return 1./(1+np.exp(-x));
randmat = lambda a, b : np.matrix(npr.rand(a, b))
zeromat = lambda a, b : np.matrix(np.zeros((a, b)))

class God:
  @staticmethod
  def gen(data, num_data):
    if isinstance(data, str):
      data = open(data, 'w')
    writer = csv.writer(data, delimiter=',')
    writer.writerow(['id']+['start.'+str(i) for i in range(H*W)]+\
                          ['stop.'+str(i) for i in range(H*W)])
    for ni in range(num_data):
      if ni%10 == 0:
        print 'ni = ', ni
      while True:
        init_density = npr.random()
        board = toInt(npr.rand(H, W) < init_density)
        (start_board, end_board) = God.evolve(board)
        if end_board != None:
          break
      writer.writerow([ni]+list(start_board.ravel('F'))+list(end_board.ravel('F')));

  @staticmethod
  def evolve_onestep(board, nextboard):
    for ni in range(board.shape[0]):
        for nj in range(board.shape[1]):
          count = 0
          for i in [ni-1, ni, ni+1]:
            for j in [nj-1, nj, nj+1]:
              count = count+board[i%H][j%W]
          if board[ni][nj] == 1 and count < 3:
            nextboard[ni][nj] = 0 # dies.
          elif board[ni][nj] == 1 and (count == 3 or count == 4):
            nextboard[ni][nj] = 1 # lives.
          elif board[ni][nj] == 1 and count > 4:
            nextboard[ni][nj] = 0 # dies.
          elif board[ni][nj] == 0 and count == 3:
            nextboard[ni][nj] = 1 # dies.
          else:
            nextboard[ni][nj] = board[ni][nj]


  @staticmethod
  def evolve(board):
    boards = [board, np.array(board)]
    start_board = None
    end_board = None
    # plt.ion()
    # delta = npr.randint(5)+1
    delta = 1
    for r in range(5+delta):
      God.evolve_onestep(boards[r%2], boards[(r+1)%2])
      if r == 4:
        start_board = boards[(r+1)%2]
    if int(sum(sum(boards[(r+1)%2]))) == 0: # empty board, discard. 
      end_board = None
    else:
      end_board = boards[(r+1)%2]
    return (start_board, end_board)
      # plt.imshow(boards[r%2])
      # plt.draw()

class Conway:
  def __init__(m, train_file, prefix, trainp = .9, num_hidden = 100, T = 20, B = 5, eta = 0.4, Q = 20, K = 5, testLag = 0.2):
    m.trainp = trainp
    m.num_hidden = num_hidden
    m.num_vis = H*W
    m.T = T
    m.B = B
    m.eta = eta
    m.Q = Q
    m.K = K
    m.prefix = prefix

    # init data.
    if isinstance(train_file, str):
      train_file = open(train_file, 'rb')
    conway_reader = csv.reader(train_file, delimiter=',')
    m.starts = list()
    m.stops = list()
    for (ri, row) in enumerate(conway_reader):
      if ri > 0:
        m.starts.append(np.matrix(toInt(row[1:H*W+1])))
        m.stops.append(np.matrix(toInt(row[H*W+1:])))
    m.num_data = len(m.starts)
    m.num_train = int(m.num_data*trainp)
    m.num_test = m.num_data-m.num_train
    m.testLag = int(m.num_train*testLag)
    m.test_starts = m.starts[m.num_train+1:]
    m.test_stops = m.stops[m.num_train+1:]
    m.starts = m.starts[:m.num_train]
    m.stops = m.stops[:m.num_train]

    # init log.
    if m.prefix != "":
      f = open(m.prefix+"log", "w")
    else:
      f = sys.stdout
    m.log("num_train_data %d" % m.num_train)
    m.log("num_test_data %d" % m.num_test)
    m.log("testlag %d" % m.testLag)
    m.log("T %d" % m.T)
    m.log("Q %d" % m.Q)
    m.log("B %d" % m.B)
    m.log("num_hidden %d" % num_hidden)
    m.log("eta %f" % m.eta)

    # init discRBM model. 
    m.weights0 = randmat(m.num_vis, m.num_vis)
    m.weights1 = randmat(m.num_hidden, m.num_vis)
    m.gradw0sq = zeromat(m.num_vis, m.num_vis)
    m.gradw1sq = zeromat(m.num_hidden, m.num_vis)

  def log(m, msg):
    if m.prefix != "":
      f = open(m.prefix+"log", "a+")
    else:
      f = sys.stdout
    f.write(msg+"\n")
    f.flush()
    f.close()

  def activateHidden(m, y):
    h = zeromat(1, m.num_hidden)
    for ni in range(m.num_hidden):
      h[0, ni] = sigmoid(m.weights1[ni, :]*y.T)
    return h

  def sampleHidden(m, y):
    h = m.activateHidden(y)
    hb = zeromat(1, m.num_hidden)
    for ni in range(m.num_hidden):
      hb[0, ni] = 1 if npr.random() < h[0, ni] else 0
    return (hb, (np.multiply(hb, 1-h)+np.multiply(1-hb, h)).T*y)

  def sampleY(m, x, h):
    y = zeromat(1, m.num_vis)
    for ni in range(m.num_vis):
      y[0, ni] = 1 if npr.random() < sigmoid(m.weights1.T[ni, :]*h.T \
                                    +m.weights0[ni, :]*x.T) else 0
    return y

  def sampleY_prior(m, h):
    y = zeromat(1, m.num_vis)
    yb = zeromat(1, m.num_vis)
    for ni in range(m.num_vis):
      y[0, ni] = sigmoid(m.weights1.T[ni, :]*h.T)
      yb[0, ni] = 1 if npr.random() < y[0, ni] else 1
    return (yb, h.T*(np.multiply(yb, 1-y)+np.multiply(1-yb, y)))

  def gradientA(m, x, y):
    h = m.activateHidden(y)
    gradw1 = h.T*y
    gradw0 = y.T*x
    y = toInt(randmat(1, m.num_vis) < .5)
    for t in range(m.T):
      (hb, _) = m.sampleHidden(y)
      y = m.sampleY(x, hb)
      if t >= m.B: # after burnin.
        gradw1 = gradw1-hb.T*y/float(m.T-m.B)
        gradw0 = gradw0-y.T*x/float(m.T-m.B)
    return (gradw0, gradw1, y, hb)

  def gradientAA(m, x, y):
    for k in range(m.K):
      gradients = list()
      weights = list()
      y = toInt(randmat(1, m.num_vis) < .5)
      for t in range(m.T):
        if t >= m.B:
          (hb, hg) = m.sampleHidden(m, y)
          gradients.append(hg)
          (yb, yg) = m.sampleY_prior(m, h)
          gradients.append(yg)
          weights.append()






  def adagrad(m, grad):
    (gradw0, gradw1, y, _) = grad
    m.gradw0sq = m.gradw0sq+np.multiply(gradw0, gradw0)
    m.gradw1sq = m.gradw1sq+np.multiply(gradw1, gradw1)
    m.weights0 = m.weights0+m.eta*np.divide(gradw0, 1e-4+np.sqrt(m.gradw0sq))
    m.weights1 = m.weights1+m.eta*np.divide(gradw1, 1e-4+np.sqrt(m.gradw1sq))
    return y

  def run(m):
    for q in range(m.Q):
      train_err = 0
      for (ni, (x, y)) in enumerate(zip(m.starts, m.stops)):
        train_y = m.adagrad(m.gradientA(x, y))
        train_err = train_err+toInt(y != train_y).sum()/float(H*W*m.testLag)
        if (q*m.num_train+ni) % m.testLag == 0:
          m.log("train err %f" % train_err)
          train_err = 0
          err = 0
          for (test_x, test_y) in zip(m.test_starts, m.test_stops):
            (_, _, y, _) = m.gradientA(test_x, test_y)
            err = err+toInt(y != test_y).sum()/float(H*W*m.num_test)
          m.log("test err %f" % err)

  def run_allzero(m):
    err = 0
    for (test_x, test_y) in zip(m.test_starts, m.test_stops):
      y = zeromat(1, m.num_vis)
      err = err+toInt(y != test_y).sum()/float(H*W*m.num_test)
    print 'all zero benchmark, err = ', err

def test_conway():
  conway = Conway("data/conway.csv")
  conway.run()

def test_conway_all_zero():
  prefix = mkprefix("allzero")
  conway = Conway("data/conway.csv", prefix)
  conway.run_allzero()

def mkprefix(name):
  num = max(0, int(os.popen('mkdir -p state/%s; cd state/%s; ls -l | wc -l'%(name, name)).read())-1)
  prefix = "state/"+name+"/%d.exec/"%num
  p = ""
  for preprefix in prefix.split('/'):
    p = p+preprefix+"/"
    if not os.path.exists(p):
      os.mkdir(p)
  return prefix

if __name__ == "__main__":
  # God.gen("data/conway.csv", 1000);
  # test_conway()
  # test_conway_all_zero()
  # sys.exit(0)

  # parse options.
  parser = OptionParser()
  parser.add_option("--name", type="string", dest="name", default="SCRATCH")
  parser.add_option("--dataset", type="string", dest="dataset",default="data/conway.csv")
  parser.add_option("--T", type="int", dest="T", default=20)
  parser.add_option("--B", type="int", dest="B", default=5)
  parser.add_option("--hidden", type="int", dest="hidden", default=100)
  parser.add_option("--Q", type="int", dest="Q", default=20)
  parser.add_option("--K", type="int", dest="K", default=5)
  parser.add_option("--testLag", type="float", dest="testLag", default=0.2)
  parser.add_option("--eta", type="float", dest="eta", default=0.4)
  (options, args) = parser.parse_args()

  # run.
  name = options.name
  prefix = mkprefix(name)
  
  conway = Conway(options.dataset, prefix, T=options.T, B=options.B, Q=options.Q, eta=options.eta,\
                   K=options.K, testLag=options.testLag, num_hidden=options.hidden)


  conway.run()


