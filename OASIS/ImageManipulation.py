from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import dia_matrix, eye
from scipy.signal import convolve2d
from scipy import stats
from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
import scipy as sp
import matplotlib.animation as animation
import time
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import Ward

def unnormalized_log_prior2(x):
    """ compute the log prior using adjacent pairs. """
    size = x.size
    z = np.zeros(x.shape)
    z[:3, :3] = 1
    z[1, 1] = 0
    z_vec = z.reshape(-1)
    non_zeros = [e - x.shape[0] - 1 for e, v in enumerate(z_vec[:x.shape[0] * 3].reshape(-1)) if v > 0]
    toep_mat = dia_matrix((size, size))
    for non_zero in non_zeros:
        toep_mat = toep_mat + eye(size, size, non_zero)
    x_flat = x.reshape(-1)
    return toep_mat.dot(x_flat)
def ising_model_update(x,t=15):
    noisy_arr_copy = x.copy()
    lmbda = 0.5
    for i in range(t):
        logodds = np.log(stats.norm.pdf(noisy_arr_copy, loc=1, scale=2)) - np.log(stats.norm.pdf(noisy_arr_copy, loc=-1, scale=2))
        noisy_arr_copy = (1 - lmbda) * noisy_arr_copy + lmbda * np.tanh(unnormalized_log_prior2(noisy_arr_copy).reshape(noisy_arr_copy.shape) + .5 * logodds)
    a = plt.imshow(noisy_arr_copy , cmap=plt.cm.gray, interpolation='nearest')
    plt.show()
    return noisy_arr_copy 
def ising_model_update_animation_ss(yy,t=15,f=15):
    class bad:
        def __init__(self):
            self.noisy_arr_copy = yy.copy()
            fig = plt.figure()
            lmbda = 0.05
            #im = plt.imshow(yy, cmap=plt.cm.gray)
            self.count = 0
            def updatefig(*args):
                if self.count > f:
                    self.noisy_arr_copy = yy.copy()
                    self.count = 0
                logodds = np.log(stats.norm.pdf(self.noisy_arr_copy, loc=1, scale=2)) - np.log(stats.norm.pdf(self.noisy_arr_copy, loc=-1, scale=2))
                self.noisy_arr_copy = (1 - lmbda) * self.noisy_arr_copy + lmbda * np.tanh(unnormalized_log_prior2(self.noisy_arr_copy).reshape(self.noisy_arr_copy.shape) + .5 * logodds)
                im.set_array(self.noisy_arr_copy)
                self.count+=1
                return im

            ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
            plt.show()
    bad()

def binary_propagation(square):
    open_square = ndimage.binary_opening(square)
    eroded_square = ndimage.binary_erosion(square)
    reconstruction = ndimage.binary_propagation(eroded_square, mask=square)
    plt.figure(figsize=(9.5, 3))
    plt.subplot(131)
    plt.imshow(square, cmap=plt.cm.hot, interpolation='nearest')
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(open_square, cmap=plt.cm.hot, interpolation='nearest')
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(reconstruction, cmap=plt.cm.hot, interpolation='nearest')
    plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0.02, top=0.99, bottom=0.01, left=0.01, right=0.99)
    plt.show()

def spect_clust_segmentation(lena,regions=20):
    X = np.reshape(lena, (-1, 1))

    connectivity = grid_to_graph(*lena.shape)

    print("Compute structured hierarchical clustering...")
    
    st = time.time()
    
    n_clusters = regions
    ward = Ward(n_clusters=n_clusters, connectivity=connectivity).fit(X)
    label = np.reshape(ward.labels_, lena.shape)
    
    print("Elapsed time: ", time.time() - st)
    print("Number of pixels: ", label.size)
    print("Number of clusters: ", np.unique(label).size)

    plt.imshow(lena, cmap=plt.cm.gray)
    for l in range(n_clusters):
        plt.contour(label == l, contours=1,
                    colors=[plt.cm.spectral(l / float(n_clusters)),])
    plt.show()
