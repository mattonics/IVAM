import numpy as np
import matplotlib.pyplot as plt
from scipy import *
from scipy.linalg import norm, pinv
import scipy.integrate as integ
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn.gaussian_process import GaussianProcess
from scipy import optimize

class RadialBasisNetwork:
     
    def __init__(self, indim, numCenters, outdim):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.centers = [random.uniform(-1, 1, indim) for i in xrange(numCenters)]
        self.beta = 8
        self.W = random.random((self.numCenters, self.outdim))
         
    def _basisfunc(self, c, d):
        assert len(d) == self.indim
        return exp(-self.beta * norm(c - d) ** 2)
     
    def _calcAct(self, X):
        G = zeros((X.shape[0], self.numCenters), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi,ci] = self._basisfunc(c, x)
        return G
     
    def fit(self, X, Y):
        rnd_idx = random.permutation(X.shape[0])[:self.numCenters]
        self.centers = [X[i,:] for i in rnd_idx]
         
        print "center", self.centers
        G = self._calcAct(X)
        print G
         
        self.W = dot(pinv(G), Y)
         
    def predict(self, X):
        G = self._calcAct(X)
        Y = dot(G, self.W)
        return Y