import numpy as np
from itertools import combinations

MAX_LOOP = 10000

def load_data(filename):
    with open(filename) as f:
        lines = [x.ljust(35, ' ').replace(' ','0').replace('#','1') for x in f.read().splitlines()]
        data = [[1 if x=='1' else -1 for x in list(s) ] for s in lines]
        return np.array(data)

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
        return (self.f(x.dot(self.w0)), self.f(x.dot(self.w)))

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

    # choose part of data
    number_chose = [1,2,4,6] # worked!
    number_chose = [1,2,3,4] # worked!
    number_chose = [3,6,7,8] # not


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
                # if not np.array_equal(pre[0], pre[1]):
                #     print c, "pre[0] != pre[1]"
                #     # print "pre[0]:", pre[0]
                #     # print "pre[1]:", pre[1]
                #     print "pre[0] - pre[1]:", pre[0]-pre[1]
