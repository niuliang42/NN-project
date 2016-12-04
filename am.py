import numpy as np
from itertools import combinations
import random

MAX_LOOP = 10000

def trans(x): # {1,0,-1,?} to "#,?, ,_"
    if x == 1: return '#'
    if x == -1: return ' '
    if x == 0: return '?'
    return '_'

def load_data(filename):
    with open(filename) as f:
        lines = [x.ljust(35, ' ').replace(' ','0').replace('#','1') for x in f.read().splitlines()]
        data = [[1 if x=='1' else -1 for x in list(s) ] for s in lines]
        return np.array(data)

def display(data):
    assert(data.size and data.size%35==0)
    data.reshape(data.size/35, 35)
    if len(data.shape)==1:
        data = np.array([data])
    for d in data:
        assert(d.shape[0]==35)
        d = [trans(e) for e in d]
        for i in range(7):
            print ''.join(d[5*i:5*i+5])
        print '-'*80

def disturbe(raw_data, n, mode = "missing"):
    """
    take (data, n, mode) as input,
    data is a (35,) np array,
    generate some noise to data, mode is 'missing' or 'mistake',
    n is the number of noisy point.
    raw data is not changed
    """
    assert(mode == "missing" or mode == "mistake")
    assert(raw_data.size == 35)
    assert(n<=35 and n>=0)
    factor = 0 if mode == "missing" else -1
    data = np.copy(raw_data)
    data = data.reshape((35,))
    chose = random.sample(range(35), n)
    for ind in chose:
        data[ind] *= factor
    return data

class AM:
    def __init__(self, X):
        self.q = np.size(X,0)
        self.n = np.size(X,1)
        self.w = np.zeros((self.n, self.n))
        self.threshold = 0
        self.w = X.T.dot(X)
        self.w0 = np.copy(self.w)
        np.fill_diagonal(self.w0, 0)
        # print "w is:\n",self.w
        # print "w0 is:\n",self.w0
        def f_notvec(x):
            if x > self.threshold:
                return 1
            elif x < self.threshold:
                return -1
            else:
                return 0
        self.f = np.vectorize(f_notvec,cache=True)

    def predict(self, x):
        """
        simply predict
        """
        return (self.f(x.dot(self.w)), self.f(x.dot(self.w0)))

    def predict_converge(self, x):
        """
        predict iteratively until converge
        """
        p0 = self.f(x.dot(self.w0))
        for i in range(MAX_LOOP): # at most iterate MAX_LOOP times
        # while True:
            p0_temp = self.f(p0.dot(self.w0))
            if np.array_equal(p0, p0_temp):
                break
            else:
                p0 = p0_temp
        p = self.f(x.dot(self.w))
        for i in range(MAX_LOOP): # at most iterate MAX_LOOP times
        # while True:
            p_temp = self.f(p.dot(self.w))
            if np.array_equal(p, p_temp):
                break
            else:
                p = p_temp
        return (p, p0)

if __name__ == "__main__":
    # load data
    train_file = "TenDigitPatterns.txt"
    train_data = load_data(train_file)
    # display(train_data)

    # choose part of data
    number_chose = [1,2,4,6] # worked!
    number_chose = [1,2,3,4] # worked!
    # number_chose = [3,6,7,8] # not


    # first question
    # if True:
    if False:
        for num in range(1,7):
            for c in combinations(range(10), num):
                # print c
                part_train_data = train_data[np.array(c)]
                am = AM(part_train_data)

                for p in [(0,am.predict_converge(part_train_data)), (1,am.predict(part_train_data))]:
                    mode = "converge" if p[0]==0 else "direct"
                    pre = p[1]
                    p0, p1= False, False
                    if np.array_equal(pre[0], part_train_data):
                        p0 = True
                    if np.array_equal(pre[1], part_train_data):
                        p1 = True
                    if p0 or p1:
                        print mode, c,
                        if p0:
                            print "p",
                        if p1:
                            print "p0",
                        print ""

    # second question
    if True:
    # if False:
        # display(train_data[0])
        # display(disturbe(train_data[0], 3, mode="missing"))
        # display(train_data[0])
        TEST_LOOP = 100
        for i in range(10):
            err = 0
            for j in range(TEST_LOOP):
                if()
                pass
            pass
