import numpy as np
from itertools import combinations

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
        while True:
            p0_temp = self.f(p0.dot(self.w0))
            if np.array_equal(p0, p0_temp):
                break
            else:
                p0 = p0_temp
        p = self.f(x.dot(self.w))
        while True:
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


    for c in combinations(range(10), 5):
        part_train_data = train_data[np.array(c)]
        # print part_train_data.shape
        # print part_train_data

        am = AM(part_train_data)
        pre = am.predict(part_train_data)
        if np.array_equal(pre[0], part_train_data):
            print c
        # print np.array_equal(pre[0], part_train_data)
        # print np.array_equal(pre[1], part_train_data)
        # print 'predict(w) == predict(w0):', np.array_equal(pre[0], pre[1])

