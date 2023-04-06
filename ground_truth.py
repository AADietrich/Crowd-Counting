import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as pil
import sklearn.neighbors as neighbors
import cv2
import scipy.io
import params
import glob

##Coef for Gaussian sample distance, determined experimentally in MCNN paper (Zhang, etal)
GAUSSIAN_BETA = 0.3
##K nearest neighbors
K = 5

image_path = params.DATA_DIR + 'train_data/images/'
truth_path = params.DATA_DIR + 'train_data/ground-truth/'
heatm_path = params.DATA_DIR + 'train_data/heatmaps/'


def getPointMap(image_path, truth_path):
    img = pil.open(image_path)
    width, height = img.size

    gt_file = scipy.io.loadmat(truth_path)
    gt = np.asarray(gt_file['image_info'][0][0][0][0][0])
    count = int(gt_file['image_info'][0][0][0][0][1])

    pointMap = np.zeros((height, width))
    for pt in gt:
        pointMap[int(pt[1]), int(pt[0])] = 1
    return pointMap, gt

def findRadius(points):
    averages = []
    nbrs = neighbors.NearestNeighbors(n_neighbors = K).fit(points)
    distances, indices = nbrs.kneighbors(points)
    for d in distances:
        averages.append(np.average(np.asarray(d[1:])))
    return np.asarray(averages)

def gaussian_old(pointMap, points):
    xis = findRadius(points)
    heatMap = np.zeros((pointMap.shape[0], pointMap.shape[1]))
    for xi in xis:
        sigma = GAUSSIAN_BETA * xi
        hm = cv2.GaussianBlur(pointMap,(0,0),sigma,sigma)
        heatMap += hm
    return heatMap

def gaussian(points, img_shape):
    xis = findRadius(points)
    heatMap = np.zeros(img_shape)
    
    for i, pt in enumerate(points):
        pointMap = np.zeros(img_shape)
        pointMap[int(pt[1]), int(pt[0])] = 1
        sigma = GAUSSIAN_BETA * xis[i]
        heatMap += cv2.GaussianBlur(pointMap, [0,0], sigma, sigma)
    return heatMap
    

images = []
truths = []
for f in glob.glob(image_path + '*'):
    images.append(f)
    truths.append(truth_path + 'GT_' + f[len(image_path):-4] + '.mat')

for i in range(len(images)):
    pointMap, points = getPointMap(images[i], truths[i])
