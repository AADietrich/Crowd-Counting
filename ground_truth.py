import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as pil
import sklearn.neighbors as neighbors


##Coef for Gaussian sample distance, determined experimentally in MCNN paper (Zhang, etal)
GAUSSIAN_BETA = 0.3
##Number nearest samples, TBD
K = 5

def findRadius(points):
    averages = []
    nbrs = neighbors.NearestNeighbors(n_neighbors = K).fit(points)
    distances, indices = nbrs.kneighbors(points)
    for d in distances:
        averages.append(np.average(np.asarray(d[1:])))
    return np.asarray(averages)

def gaussian(points, xis):
    for i in xis:
        
