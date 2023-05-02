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

class PreProcess():
    def __init__(self, mode):
        self.image_path = p.DATA_DIR + mode + '_data/images/'
        self.truth_path = p.DATA_DIR + mode + '_data/ground-truth/'
        self.heatm_path = p.DATA_DIR + mode + '_data/heatmaps/'        
        return
    
    def generate_hms(self):
        images = []
        truths = []


        for f in glob.glob(self.image_path + '*'):
            images.append(f)
            truths.append(self.truth_path + 'GT_' + f[len(self.image_path):-4] + '.mat')
        
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
            hmImg.save(self.heatm_path + images[i][len(self.image_path):-4] + "_hm.png")
            print("Processed " + str(i) + " images out of " + str(len(truths)), end='\r')  
        return

    def crop_images(self):
        with open('.//counts.csv', 'w') as f:
            writer = csv.writer(f)
            for f in glob.glob(self.image_path + '*.jpg'):
                name = f[len(self.image_path):-4]
                count = crop.crop(self.image_path + name + '.jpg', self.heatm_path + name + '_hm.png', self.truth_path + 'GT_' + name + '.mat')
                writer.writerow([name, count])
        return

    def flip_images(self):
        for f in glob.glob(self.image_path + '*_cropped.png'):
            #Flip 30% of images L-R at random (augmentation proc used for Shanghaitech data in MRC paper)
            r = random.randint(1,100)
            if(r > 70):
                name = f[len(self.image_path):-12]
                img = pil.open(f)
                hm = pil.open(self.heatm_path + name + '_hm_cropped.png')
                img = pilio.mirror(img)
                hm = pilio.mirror(hm)
                img.save(f)
                hm.save(self.heatm_path + name + '_hm_cropped.png')
        return
            
    def load_data(self):
        xarr = []
        yarr = []
        for f in glob.glob(self.image_path + '*_cropped.png'):
            x = np.asarray(pil.open(f))
            #Rescale to -1,1
            x = (x - 127.5)/128
            xarr.append(x)
        for f in glob.glob(self.heatm_path + '*_cropped.png'):
            y = np.asarray(pil.open(f))
            #Rescale to -1,1
            y = (y - 127.5)/128
            yarr.append(x)
        npx = np.stack(xarr,axis=0)
        npy = np.stack(yarr,axis=0)
        return npx, npy