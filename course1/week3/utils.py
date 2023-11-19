import numpy as np

def load_data(path):
    data = np.loadtxt(path, delimiter=',')
    X = data[:,:2]
    y = data[:,2]
    return X, y
