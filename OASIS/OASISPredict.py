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


if  __name__ == "__main__":
    base_string = 'D:\MriData\Data'
    excel_path = 'D:\oasis_cross-sectional.csv'
    test = 13
    dataprovider = CrossSectionalData.CrossSectionalDataProvider(base_string,excel_path)

    a = dataprovider.get_data_with_CDR()

    print(dataprovider.get_CDR(test)) 

    step = 5
    step_1 = 20
    step_2 = 25

    
    training_stop = int(len(a)/2)
    allfeatures = []
    ally = []
    cut = 55
    randomint = random.Random(7)
    for xx in [randomint.randint(0,len(a) - 1) for r in xrange(training_stop)]:
        x = a[xx]
        cdr = dataprovider.get_CDR(x)
        print(cdr)
        ll = (dataprovider.retrieve_full_data(x))
        #AlzheimerFeatures.view_histogram(ll)
        #CrossSectionalData.show_slices([ll[:,:,50]])
        if cdr == None or cdr > 1:
            continue
        feat = AlzheimerFeatures.surrounding_points_discrete_with_pos(ll,step,step_1,[dataprovider.get_gender(x)])
        allfeatures+=feat
        ally = np.append(ally,np.repeat(cdr,len(feat)))

    AlzheimerFeatures.shuffle_in_unison_scary(allfeatures,ally)

    regressor = sk.ExtraTreeRegressor(random_state=0)
    regressor.fit(allfeatures,ally)

    net = GaussianNB()
    net.fit(np.array(allfeatures),np.array(ally))

    def f(x):
        if x == 0.5:
            return 0
        if x == 0:
            return -1
        return 1

    ttt = AlzheimerFeatures.target_brain_regions_2d_z(dataprovider.retrieve_full_data(test),step,[dataprovider.get_gender(test)],lambda x: f(net.predict(x)),50)
    print(im.ising_model_update_animation(ttt[:,:,50]))

    allfeatures1 = []
    ally1 = []
    indices = []
    for xx in [randomint.randint(0,len(a) - 1) for r in xrange(training_stop) if not r == test]:
        x = a[xx]
        indices.append(xx)
        cdr = dataprovider.get_CDR(x)
        ll = dataprovider.retrieve_full_data(x)
        #CrossSectionalData.show_slices([ll[:,:,50]])
        if cdr == None or cdr > 1:
            continue
        feat = AlzheimerFeatures.surrounding_points_discrete_with_pos(ll,step,step_2,[dataprovider.get_gender(x)])
        print(len(feat))
        allfeatures1.append(regressor.predict(feat)[0:cut])
        #plt.plot(regressor.predict(feat))
        #plt.show()
        ally1.append(cdr)
    
    rbf_svc = neighbors.KNeighborsClassifier(n_neighbors = 7)
    rbf_svc.fit(allfeatures1, ally1)

    ll = dataprovider.retrieve_full_data(56)
    error = 0
    index = 0
    for xx in range(len(a)):
        x = a[xx]
        cdr = dataprovider.get_CDR(x)
        if cdr == None or cdr > 1 or xx in indices:
            continue
        ll = dataprovider.retrieve_full_data(x)
        feat = AlzheimerFeatures.surrounding_points_discrete_with_pos(ll,step,step_2,[dataprovider.get_gender(x)])
        predictq = regressor.predict(feat)[:cut]
        #plt.plot(predict)
        #plt.show()
        suma = (rbf_svc.predict(predictq))
        #print(suma)
        error += np.abs(suma - cdr)
        index += 1
        print(1 - (error / index))
    print(1 - (error / index))


