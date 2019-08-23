import numpy as np
import matplotlib.pyplot as plt

"""AUC of kernel should be 1"""
def kernel_density_probability_estimator(data,kernel,smoothing=None):
    n = len(data)
    if not smoothing:
        smoothing = 1.06 * np.std(data) * np.power(n,-0.2)
    normal = 1.0 / (n * smoothing)
    return lambda x : _sum(data,x,kernel,normal,smoothing)

def kernel_density_visualization(data,kernel,smoothing=None,growth=0.01,title=None):
    f = kernel_density_probability_estimator(data,kernel,smoothing)
    a = np.arange(min(data),max(data),growth)
    if title:
        plt.title(title)
    plt.plot(a, [f(x) for x in a])
    plt.show()
    return f
def gaussian_kernel(x):
    l = 1.0 / np.sqrt(2 * np.pi)
    a = np.exp(-0.5 * (x ** 2))
    return l * a
def _sum(data,x,kernel,normal,h):
    sum = 0
    for i in data:
        sum+=kernel((x - i) / h)
    return normal * sum

class CustomOutlierDetection:
    def __init__(self,entitys,attributes,attributedata,kernel,outlier_thres=0.05):
        self.outlier_thres = outlier_thres
        self.length = len(attributedata)

        self.entitymap = {}
        for r in range(len(entitys)):
            self.entitymap[entitys[r]] = r

        self.attributemap = {}
        for r in range(len(attributes)):
            self.attributemap[r] = attributes[r]

        self.data = attributedata
        self.distributions = [kernel_density_probability_estimator(self.data[:,x],kernel,None) for x in range(self.length)]
    
    def find_outlier_for_entity(self,st):
        attributes = []
        l = self.entitymap[st]
        for x in range(self.length):
            calc = self.distributions[x](self.data[l,x])
            if calc < self.outlier_thres:
                attributes.append((self.attributemap[x],calc))
        return attributes
