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
import time
import scipy.stats as stats
import nibabel as nb


if  __name__ == "__main__":
    base_string = 'D:\MriData\Data'
    excel_path = 'D:\oasis_cross-sectional.csv'
    test = 13
    dataprovider = CrossSectionalData.CrossSectionalDataProvider(base_string,excel_path,False)

    a = dataprovider.get_data_with_CDR()

    print(dataprovider.get_CDR(test)) 

    step = 6
    step_1 = 25
    step_2 = 25

    
    training_stop = int(len(a))
    allfeatures = []
    ally = []
    randomint = random.Random(7)
    for xx in xrange(len(a)):
        x = a[xx]
        cdr = dataprovider.get_CDR(x)
        ll = (dataprovider.retrieve_full_data(x))
        if cdr == None or cdr > 1:
            continue
        feat = AlzheimerFeatures.surrounding_points_discrete_with_pos(ll,step,step_1,[dataprovider.get_gender(x)])
        print(len(feat))
        allfeatures+=feat
        ally = np.append(ally,np.repeat(cdr,len(feat)))

    AlzheimerFeatures.shuffle_in_unison_scary(allfeatures,ally)

    net = GaussianNB()
    net.fit(np.array(allfeatures),np.array(ally))

    def f(x):
        if x == 0.5:
            return 0
        if x == 0:
            return -1
        return 1
    slice = 50
    
    ttt = AlzheimerFeatures.target_brain_regions_2d_z(dataprovider.retrieve_full_data(test),step,[dataprovider.get_gender(test)],lambda x: f(net.predict(x)),slice)
    (im.ising_model_update_animation_ss(ttt[:,:,slice],1))
    
