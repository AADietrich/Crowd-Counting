import glob
import scipy.io
import numpy as np
import PIL.Image as pil
import PIL.ImageOps as pilio
import random
import csv
import cv2

import heatmap
import crop
import params as p

class PreProcess():
    def __init__(self, mode):
        self.mode = mode
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
            
        print("Heatmap generation of " + self.mode + " data complete.")
        return

    def crop_images(self):
        #Write counts to csv to use later
        with open('.//counts_' + self.mode + '.csv', 'w') as f:
            writer = csv.writer(f)
            for f in glob.glob(self.image_path + '*.jpg'):
                name = f[len(self.image_path):-4]
                count = crop.crop(self.image_path + name + '.jpg', self.heatm_path + name + '_hm.png', self.truth_path + 'GT_' + name + '.mat')
                writer.writerow([name, count])
        print("Cropping of " + self.mode + " data complete.")
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
        print("Flipping of " + self.mode + " data complete.")
        return
    
    def downsample(self):
        for f in glob.glob(self.heatm_path + '*_cropped.png'):
            img = cv2.imread(f)
            #Use 2 pyramids to downsample to 1/4 size
            img = cv2.pyrDown(img)
            img = cv2.pyrDown(img)
            cv2.imwrite(f[:-12] + '_cropdown.png', img)
        print("Downsampling of " + self.mode + " data complete.")
        return        
    
    def load_data(self):
        xarr = []
        yarr1 = []
        yarr2 = []
        for f in glob.glob(self.image_path + '*_cropped.png'):
            x = np.asarray(pil.open(f))
            #Rescale to -1,1
            x = (x - 127.5)/128
            xarr.append(x)
        for f in glob.glob(self.heatm_path + '*_cropdown.png'):
            y1 = np.asarray(pil.open(f))
            #Rescale to -1,1
            y1 = (y1 - 127.5)/128
            yarr1.append(y1)
        for f in glob.glob(self.heatm_path + '*_cropped.png'):
            y2 = np.asarray(pil.open(f))
            #Rescale to -1,1
            y2 = (y2 - 127.5)/128
            yarr2.append(y2)
        npx = np.stack(xarr,axis=0)
        npy1 = np.stack(yarr1,axis=0)
        npy2 = np.stack(yarr2,axis=0)
        print(self.mode + " data loaded")
        return npx, npy1, npy2