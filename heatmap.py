import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as pil
import sklearn.neighbors as neighbors
import cv2
import params as p
import glob

#Coef for Gaussian sample distance, determined experimentally in MCNN paper (Zhang, etal)
GAUSSIAN_BETA = 0.3

#K nearest neighbors to determine head radius
K = 5

def findRadius(points):
    averages = []
    nbrs = neighbors.NearestNeighbors(n_neighbors = K).fit(points)
    distances, indices = nbrs.kneighbors(points)
    for d in distances:
        averages.append(np.average(np.asarray(d[1:])))
    return np.asarray(averages)

def gaussian(points, img_shape):
    xis = findRadius(points)
    heatMap = np.zeros(img_shape)
    
    #Create point map for each head location (delta-dirac representation)
    #Add gaussian blur for each pointmap to heatmap
    for i, pt in enumerate(points):
        pointMap = np.zeros(img_shape)
        pointMap[int(pt[1]), int(pt[0])] = 1
        sigma = GAUSSIAN_BETA * xis[i]
        heatMap += cv2.GaussianBlur(pointMap, [0,0], sigma, sigma)
    return heatMap
