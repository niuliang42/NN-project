import numpy as np
from itertools import combinations
import random

MAX_LOOP = 10000 # parameter for AM.predict_converge(): maximum loop times
TEST_LOOP = 100 # parameter for Q2: for a err number, try homw many times
ERR_RANGE = 35   # parameter for Q2: the domain of err number
VERBOSE = False  # True to output more details

def trans(x): # turn {1,0,-1,?} into {#,?, ,_}
    if x == 1: return '#'
    if x == -1: return ' '
    if x == 0: return '?'
    return '_'

def load_data(filename): # load data from txt file
    with open(filename) as f:
        lines = [x.ljust(35, ' ').replace(' ','0').replace('#','1') for x in f.read().splitlines()]
        data = [[1 if x=='1' else -1 for x in list(s) ] for s in lines]
        return np.array(data)

def display(data): # visualize a pattern(a 35x1 array)
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
            p0_temp = self.f(p0.dot(self.w0))
            if np.array_equal(p0, p0_temp):
                break
            else:
                p0 = p0_temp
        p = self.f(x.dot(self.w))
        for i in range(MAX_LOOP): # at most iterate MAX_LOOP times
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
    if VERBOSE:
        display(train_data)

    ##################
    # first question #
    ##################
    good_c = []
    if True:
    # if False:
        print "*"*80
        print "# first question #"
        for num in range(1,8):
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
                        good_c.append(c)
                        if p0:
                            print "p",
                        if p1:
                            print "p0",
                        print ""
        print good_c

    ###################
    # second question #
    ###################
    if True:
    # if False:
        print "*"*80
        print "# second question #"
        for num in range(1,8):
            for c in combinations(range(10), num):
                if c in good_c:
                    print c
                    part_train_data = train_data[np.array(c)]
                    am = AM(part_train_data)

                    D_N = part_train_data.shape[0]
                    for err_n in range(ERR_RANGE): # error number
                        err_missing = 0.0
                        err_mistake = 0.0
                        for i in range(TEST_LOOP):
                            for d in part_train_data:
                                missing_d= disturbe(d, err_n, mode="missing")
                                mistake_d= disturbe(d, err_n, mode="mistake")
                                missing_pre = am.predict(missing_d)[1] # using w0 to predict
                                mistake_pre = am.predict(mistake_d)[1] # using w0 to predict
                                if VERBOSE:
                                    display(d)
                                    display(missing_d)
                                    display(mistake_d)
                                    display(missing_pre)
                                    display(mistake_pre)
                                if not np.array_equal(missing_pre, d):
                                    err_missing += 1
                                if not np.array_equal(mistake_pre, d):
                                    err_mistake += 1
                        err_rate_missing = err_missing/(TEST_LOOP * D_N)
                        err_rate_mistake = err_mistake/(TEST_LOOP * D_N)
                        print "When err is", err_n, ", missing ERR rate:", err_rate_missing, ", mistake ERR rate:", err_rate_mistake

    ##################
    # Third question #
    ##################
    if True:
    # if False:
        print "*"*80
        print "# second question #"
        def spurious(pre, train_data): # given a predict, judge if it's in the stored patterns
            for d in train_data:
                if np.array_equal(d, pre):
                    return False
            return True
        # c= (1, 4, 6, 7, 9)
        # part_train_data = train_data[np.array(c)]
        # am = AM(part_train_data)
        set_p = set()
        total_n = 0

        # find out the spurious patterns
        for num in range(1,8):
            for c in combinations(range(10), num):
                if c in good_c:
                    # print c
                    part_train_data = train_data[np.array(c)]
                    am = AM(part_train_data)
                    # pre = am.predict_converge(part_train_data)[0]
                    pre = am.predict(part_train_data)[1] # using w0 to predict
                    total_n += pre.shape[0]
                    for p in pre:
                        p.reshape(35,)
                        if spurious(p, part_train_data):
                            t = tuple(p)
                            set_p.add(t)

        # show these spurious patterns
        for sp in set_p:
            display(np.array(sp))
        print len(set_p)
        print total_n

    print "*"*80
