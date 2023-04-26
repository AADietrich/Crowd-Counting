import ground_truth
import crop

images = []
truths = []
for f in glob.glob(image_path + '*'):
    images.append(f)
    truths.append(truth_path + 'GT_' + f[len(image_path):-4] + '.mat')

for i in range(len(images)):
    pointMap, points = getPointMap(images[i], truths[i])
