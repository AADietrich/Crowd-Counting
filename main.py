import network
import preprocess
import params as p



#preprocess.generate_hms()
#preprocess.crop_images()
#preprocess.flip_images()
x_train, y_train = preprocess.get_patches()


net = network.MRC()
#net.buildModel()
net.loadModel()
net.train(x_train,y_train)
