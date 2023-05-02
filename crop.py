import random
import PIL.Image as pil
import params as p
import scipy.io
import numpy as np


def crop(image_path, heatmap_path, truth_path):
    image = pil.open(image_path)
    heatmap = pil.open(heatmap_path)
    w,h = image.size
    x,y = p.X, p.Y
    
    #Discard image if smaller than the patch size
    if(x>=w or y>=h):
        return
    hm = np.zeros((x,y))
    #Crop image and heatmap until patch with >1 heads found
    while(hm.max() == 0):
        left = random.randint(0,w-x-1)
        top = random.randint(0,h-y-1)
        cropped_image = image.crop((left, top, left+x, top+y))
        cropped_heatmap = heatmap.crop((left, top, left+x, top+y))
        hm = np.asarray(cropped_heatmap)
            
    #Renormalize heatmap patch before saving
    hm = hm*(1.0/hm.max())
    hmImg = pil.fromarray(np.uint8(hm*255))
    hmImg.convert('L').save(heatmap_path[:-4] + "_cropped.png")
    cropped_image.convert('L').save(image_path[:-4] + "_cropped.png")
    
    #Now get count of people in patch from original gt file
    count = 0
    gt_file = scipy.io.loadmat(truth_path)
    gt = np.asarray(gt_file['image_info'][0][0][0][0][0])
    for t in gt:
        if(left < t[0] < left+x and top < t[1] < top+y):
            count += 1
    
    return count
