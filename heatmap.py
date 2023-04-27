import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as pil
import sklearn.neighbors as neighbors
import cv2
import params as p
import glob
"""
def loadImage(image_path, truth_path):
    img = pil.open(image_path)
    width, height = img.size

    gt_file = scipy.io.loadmat(truth_path)
    gt = np.asarray(gt_file['image_info'][0][0][0][0][0])
    count = int(gt_file['image_info'][0][0][0][0][1])

    return gt, count
"""
def heatmap():
def findRadius(points):
    averages = []
    nbrs = neighbors.NearestNeighbors(n_neighbors = p.K).fit(points)
    distances, indices = nbrs.kneighbors(points)
    for d in distances:
        averages.append(np.average(np.asarray(d[1:])))
    return np.asarray(averages)

def gaussian(points, img_shape):
    xis = findRadius(points)
    heatMap = np.zeros(img_shape)
    
    for i, pt in enumerate(points):
        pointMap = np.zeros(img_shape)
        pointMap[int(pt[1]), int(pt[0])] = 1
        sigma = p.GAUSSIAN_BETA * xis[i]
        heatMap += cv2.GaussianBlur(pointMap, [0,0], sigma, sigma)
    return heatMap
