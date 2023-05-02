import glob
import scipy.io
import numpy as np
import PIL.Image as pil
import PIL.ImageOps as pilio
import random
import csv

import heatmap
import crop
import params as p

IMAGE_PATH = p.DATA_DIR + 'train_data/images/'
TRUTH_PATH = p.DATA_DIR + 'train_data/ground-truth/'
HEATM_PATH = p.DATA_DIR + 'train_data/heatmaps/'

def generate_hms():
    images = []
    truths = []


    for f in glob.glob(IMAGE_PATH + '*'):
        images.append(f)
        truths.append(TRUTH_PATH + 'GT_' + f[len(IMAGE_PATH):-4] + '.mat')
    
    #counts = []
    for i in range(len(truths)):
        img = pil.open(images[i])
        width,height = img.size
        img.close()
        
        gt_file = scipy.io.loadmat(truths[i])
        gt = np.asarray(gt_file['image_info'][0][0][0][0][0])
        #count = int(gt_file['image_info'][0][0][0][0][1])
        
        hm = heatmap.gaussian(gt,(height,width))
        #Normalize gaussian heatmap to 8 bit greyscale
        hm = hm*(1.0/hm.max())
        hmImg = pil.fromarray(np.uint8(hm*255))
        hmImg.save(HEATM_PATH + images[i][len(IMAGE_PATH):-4] + "_hm.png")
        print("Processed " + str(i) + " images out of " + str(len(truths)), end='\r')  
    return

def crop_images():
    with open('.//counts.csv', 'w') as f:
        writer = csv.writer(f)
        for f in glob.glob(IMAGE_PATH + '*.jpg'):
            name = f[len(IMAGE_PATH):-4]
            count = crop.crop(IMAGE_PATH + name + '.jpg', HEATM_PATH + name + '_hm.png', TRUTH_PATH + 'GT_' + name + '.mat')
            writer.writerow([name, count])
    return

#Flip 30% of images L-R at random (augmentation proc used for Shanghaitech data in MRC paper)
def flip_images():
    for f in glob.glob(IMAGE_PATH + '*_cropped.png'):
        r = random.randint(1,100)
        if(r > 70):
            name = f[len(IMAGE_PATH):-12]
            img = pil.open(f)
            hm = pil.open(HEATM_PATH + name + '_hm_cropped.png')
            img = pilio.mirror(img)
            hm = pilio.mirror(hm)
            img.save(f)
            hm.save(HEATM_PATH + name + '_hm_cropped.png')
    return
        
def get_patches():
    xarr = []
    yarr = []
    for f in glob.glob(IMAGE_PATH + '*_cropped.png'):
        x = np.asarray(pil.open(f))
        #Rescale to -1,1
        x = (x - 127.5)/128
        xarr.append(x)
    for f in glob.glob(HEATM_PATH + '*_cropped.png'):
        y = np.asarray(pil.open(f))
        #Rescale to -1,1
        y = (y - 127.5)/128
        yarr.append(x)
    npx = np.stack(xarr,axis=0)
    npy = np.stack(yarr,axis=0)
    return npx, npy