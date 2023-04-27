import glob
import scipy.io
import numpy as np
import PIL.Image as pil

import heatmap
import crop
import params as p

def process(mode):
    images = []
    truths = []


    for f in glob.glob(p.IMAGE_PATH + '*'):
        images.append(f)
        truths.append(p.TRUTH_PATH + 'GT_' + f[len(p.IMAGE_PATH):-4] + '.mat')
    
    counts = []
    for i in range(len(truths)):
        img = pil.open(images[i])
        width,height = img.size
        img.close()
        
        gt_file = scipy.io.loadmat(truths[i])
        gt = np.asarray(gt_file['image_info'][0][0][0][0][0])
        count = int(gt_file['image_info'][0][0][0][0][1])
        
        hm = heatmap.gaussian(gt,(height,width))
        hm = hm*(1.0/hm.max())
        hmImg = pil.fromarray(np.uint8(hm*255))
        hmImg.save(p.HEATM_PATH + images[i][len(p.IMAGE_PATH):-4] + "_hm.png")
        print("Processed " + str(i) + " images out of " + str(len(truths)), end='\r')  
    return

process()