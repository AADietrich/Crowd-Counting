import network
import preprocess

'''Training data preprocessing'''
train = preprocess.PreProcess("train")
#train.generate_hms()
#train.crop_images()
#train.flip_images()
#train.downsample()
x_train, y1_train, y2_train = train.load_data()

'''Testing data preprocessing'''
test = preprocess.PreProcess("test")
#test.generate_hms()
#test.crop_images()
#test.flip_images()
#test.downsample()
x_test, y1_test, y2_test = test.load_data()

'''Train/test'''
net = network.MRC()
#net.buildModel()
net.loadModel()
net.train(x_train, y1_train, y2_train)
net.saveModel()
net.test(x_test, y1_test, y2_test)
