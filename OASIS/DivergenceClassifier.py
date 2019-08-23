import numpy as np
import scipy.stats as stats
class DivergenceClassifier():
    def __init__(self):
        None
    def fit(self,X,y):
        self.data = X
        self.predictval = y
        self.max = []
        self.lendata = len(self.data)
        for x in range(self.lendata):
            self.max.append(np.max(self.data[x]))
        return self
    def predict(self,xx):
        lenk = len(xx)
        min = 0
        k = -1
        for x in range(self.lendata):
            min = self._min(lenk,len(self.data[x]))
            L = np.histogram(xx[:min],normed = True)
            val = stats.entropy(L[1], np.histogram(self.data[x][:min])[1])
            if val < min:
                k = x
                min = val
        return self.predictval[k]
    def predict_list(self,xx):
        a = [self.predict(x) for x in xx]
        return a
    def _min(self,x,y):
        if x > y:
            return y
        return x