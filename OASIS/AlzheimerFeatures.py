import numpy as np
import matplotlib.pyplot as plt
import pybrain.datasets as ds
from numpy import fft

def surrounding_points_smooth(slice,k=10):
    return surrounding_points_discrete_with_pos(slice,k,1)

def surrounding_points_discrete_with_pos(slice,k=10,step=None,add=None):
    shape = slice.shape
    step = step or k
    lenx = shape[0]
    leny = shape[1]
    lenz = shape[2]
    half = k / 2
    lista = []

    for x in range(k,lenx - half,step):
        for y in range(k,leny - half,step):
            for z in range(k,lenz - half,step):
                data = np.reshape(slice[x - half:x + half,y - half:y + half,z - half:z + half],-1)
                if np.all(data == 0):
                    continue
                if add:
                    data = np.append(data,add)
                lista.append(np.append(data,[x,y,z]))
    return lista
def surrounding_points_discrete_with_pos_show(slice,f,k=10,step=2):
    shape = slice.shape
    step = step or k
    lenx = shape[0]
    leny = shape[1]
    lenz = shape[2]
    
    lista = np.zeros((int(lenx - k) + 1,int(leny - k) + 1,int(lenz - k) + 1))

    for x in range(k,lenx,step):
        print(x)
        for y in range(k,leny,step):
            for z in range(k,lenz,step):
                data = np.reshape(slice[x - k:x,y - k:y,z - k:z],-1)
                try:
                    lista[x - k,y - k, z - k] = f(np.append(data,[x,y,z]))
                except:
                    continue
    return lista

def target_brain_regions(slice,k,add,model):    
    shape = slice.shape

    lenx = shape[0]
    leny = shape[1]
    lenz = shape[2]
    half = k / 2
    
    lista = np.zeros(shape)

    for x in range(k,lenx - half,1):
        print(x)
        for y in range(k,leny - half,1):
            def inner_loop(z):
                data = np.reshape(slice[x - half:x + half,y - half:y + half,z - half:z + half],-1)
                if np.sum(data) == 0:
                    lista[x,y,z] = 0
                    return
                if add:
                    data = np.append(data,add)
                lista[x,y,z] = model(np.append(data,[x,y,z]))     

            map(inner_loop,xrange(half,lenz - half))           
    return np.array(lista)

def target_brain_regions_2d_z(slice,k,add,model,z):    
    shape = slice.shape

    lenx = shape[0]
    leny = shape[1]

    lista = np.zeros(shape)
    half = k / 2
    for x in range(half,lenx - half):
        print(x)
        for y in range(leny - half):
            def inner_loop(z):
                data = np.reshape(slice[x - half:x + half,y - half:y + half,z - half:z + half],-1)
                if np.sum(data) == 0:
                    lista[x,y,z] = -2
                    return
                if add:
                    data = np.append(data,add)
                lista[x,y,z] = model(np.append(data,[x,y,z]))     
                #print(lista[x,y,z])
            inner_loop(z)           
    return np.array(lista)
def target_brain_regions_2d_y(slice,k,add,model,y):    
    shape = slice.shape

    lenx = shape[0]
    leny = shape[1]
    lenz = shape[2]
    lista = np.zeros(shape)
    half = k / 2
    
    for x in range(0,lenx):
        print(x)
        def inner_loop(z):
            data = np.reshape(slice[x - half:x + half,y - half:y + half,z - half:z + half],-1)
            if np.sum(data) == 0:
                lista[x,y,z] = -2
                return
            if add:
                data = np.append(data,add)
            lista[x,y,z] = model(np.append(data,[x,y,z]))     
        map(inner_loop,xrange(lenz))           
    return np.array(lista)
def view_histogram(slice,k=1,step=5,add=None):
    shape = slice.shape
    step = step or k
    lenx = shape[0]
    leny = shape[1]
    lenz = shape[2]
    
    lista = []

    for x in range(k,lenx,step):
        print(x)
        for y in range(k,leny,step):
            for z in range(k,lenz,step):
                data = np.reshape(slice[x - k:x,y - k:y,z - k:z],-1)
                if np.all(data == 0):
                    continue
                if add:
                    data = np.append(data,add)
                lista.append(np.append(data,[x,y,z]))
    lista = fft.fft(np.reshape(lista,-1))
    plt.plot(lista)
    plt.show()
def to_supervised_dataset(list,target):
    super = ds.SupervisedDataSet(len(list[0]),1)
    for l in enumerate(list):
        super.addSample(l,target)
def add_to_supervised_dataset(sp,data,target):
    for l in range(len(data)):
        sp.addSample(data[l],target)
def round_to_class(feat,list):
    min = 100000000
    p = None
    for i in range(len(list)):
        k = abs(feat - list[i])
        if k < min:
            p = list[i]
            min = k
    return p


def shuffle_in_unison_scary(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)