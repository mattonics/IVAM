import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import CrossSectionalData
import AlzheimerFeatures
import sklearn.tree as sk
from sklearn import svm
import sklearn.ensemble as en
import DivergenceClassifier as dc
from sklearn import neighbors, datasets
import random
import scipy.ndimage.filters as filters
from sklearn.neighbors.nearest_centroid import NearestCentroid
import sklearn.metrics.pairwise as metric
import ImageManipulation as im
import RBFN as rbfn
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcess
import KernelDensityEstimation as kde


def run(perc):
    base_string = 'D:\MriData\Data'
    excel_path = 'D:\oasis_cross-sectional.csv'
    test = 13
    dataprovider = CrossSectionalData.CrossSectionalDataProvider(base_string,excel_path)

    a = dataprovider.get_data_with_CDR()

    step = 5
    step_1 = 25
    step_2 = 25

    
    training_stop = int(len(a) * perc)
    allfeatures = []
    ally = []
    cut = 55
    randomint = random.Random(7)
    for xx in [randomint.randint(0,len(a) - 1) for r in xrange(training_stop)]:
        x = a[xx]
        cdr = dataprovider.get_CDR(x)
        ll = (dataprovider.retrieve_full_data(x))
        if cdr == None or cdr > 1:
            continue
        feat = AlzheimerFeatures.surrounding_points_discrete_with_pos(ll,step,step_1,[dataprovider.get_gender(x)])
        allfeatures+=feat
        ally = np.append(ally,np.repeat(cdr,len(feat)))

    AlzheimerFeatures.shuffle_in_unison_scary(allfeatures,ally)

    regressor = sk.ExtraTreeRegressor(random_state=0)
    regressor.fit(allfeatures,ally)

    allfeatures1 = []
    ally1 = []
    indices = []
    for xx in [randomint.randint(0,len(a) - 1) for r in xrange(training_stop) if not r == test]:
        x = a[xx]
        indices.append(xx)
        cdr = dataprovider.get_CDR(x)
        ll = dataprovider.retrieve_full_data(x)
        if cdr == None or cdr > 1:
            continue
        feat = AlzheimerFeatures.surrounding_points_discrete_with_pos(ll,step,step_2,[dataprovider.get_gender(x)])
        allfeatures1.append(regressor.predict(feat)[0:cut])
        ally1.append(cdr)
    
    rbf_svc = neighbors.KNeighborsClassifier(n_neighbors = 7)
    rbf_svc.fit(allfeatures1, ally1)

    errorb = 0
    error = 0
    index = 0
    for xx in xrange(len(a)):
        x = a[xx]
        cdr = dataprovider.get_CDR(x)
        if cdr == None or cdr > 1 or xx in indices:
            continue
        ll = dataprovider.retrieve_full_data(x)
        feat = AlzheimerFeatures.surrounding_points_discrete_with_pos(ll,step,step_2,[dataprovider.get_gender(x)])
        predictq = regressor.predict(feat)[:cut]
        suma = (rbf_svc.predict(predictq))
        if not (suma > 0 and cdr > 0) or suma == cdr:
            errorb+=1
        error += np.abs(suma - cdr)
        index += 1
    ter = 1 - (error / index)
    terb = 1 - (errorb / index)
    print(str(ter) + " , " + str(terb))
    return ter,terb

list = []
for x in np.arange(.5,.9,.02):
    regress,binary = run(x) 
    list.append([regress,binary])
np.savetxt("dft.csv",list,delimiter=',')

