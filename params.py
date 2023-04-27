##################################################################################################
#FILEPATHS
##################################################################################################
DATA_DIR = '../ShanghaiTech/part_A/'

IMAGE_PATH = DATA_DIR + 'train_data/images/'
TRUTH_PATH = DATA_DIR + 'train_data/ground-truth/'
HEATM_PATH = DATA_DIR + 'train_data/heatmaps/'

##################################################################################################
#PREPROCESSING CONSTANTS
##################################################################################################
PATCH_X = 224
PATCH_Y = 224

#Coef for Gaussian sample distance, determined experimentally in MCNN paper (Zhang, etal)
GAUSSIAN_BETA = 0.3

#K nearest neighbors to determine head radius
K = 5
