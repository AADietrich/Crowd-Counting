import numpy as np
import PIL.Image as pil
import tensorflow as tf
from tensorflow import keras 
from keras import layers
import glob

import params as p

LEARN_RATE = 0.000003
LH_WEIGHT = 0.0001
BATCH_SIZE = 40
EPOCHS = 10

PATCH_PATH = p.DATA_DIR + 'train_data/patches/'
SAVE_PATH = './models'

class MRC:
    def __init__(self):
        self.model = self.buildModel()
        self.model.compile(loss=keras.losses.MeanSquaredError(),
                      optimizer=keras.optimizers.Adam(learning_rate=LEARN_RATE),
                      metrics=["mse"]  )
        return
    def train(self,x_train, y_train):
        history = self.model.fit(x_train,y_train,batch_size=BATCH_SIZE,epochs=EPOCHS,validation_split=0.2)
                
        return
    def compileModel():
        """
        model.compile(
            loss={"o1": "mse",
                    "o2": "mse"},
            loss_weights={"o1": 1.0,
                        "o2": LH_WEIGHT},
            optimizer=keras.optimizers.Adam(learning_rate=LEARN_RATE),
            metrics=["accuracy"]       
        """
        return
    
    def buildModel(self):
        #temp image dimensions - must be multiple of 32 (2 sampling ratio ^ 5 blocks) so that dimensions are correct for lateral connections
        w = p.X
        h = p.Y

        #Define model input
        inputs = keras.Input(shape=(w,h,1))

        print(inputs.shape)
        print(inputs.dtype)

        # x used for main sequential path through model, s# used for lateral connections, o# for outputs

        #Encoder

        #First VGG block
        x = layers.Conv2D(64,3,activation="relu",padding="same")(inputs)
        x = layers.Conv2D(64,3,activation="relu",padding="same")(x)
        x = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(x)


        #Second block
        x = layers.Conv2D(128,3,activation="relu",padding="same")(x)
        s1 = layers.Conv2D(128,3,activation="relu",padding="same")(x)
        x = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(s1)


        #Third block
        x = layers.Conv2D(256,3,activation="relu",padding="same")(x)
        x = layers.Conv2D(256,3,activation="relu",padding="same")(x)
        s2 = layers.Conv2D(256,3,activation="relu",padding="same")(x)
        x = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(s2)


        #Fourth block
        x = layers.Conv2D(512,3,activation="relu",padding="same")(x)
        x = layers.Conv2D(512,3,activation="relu",padding="same")(x)
        s3 = layers.Conv2D(512,3,activation="relu",padding="same")(x)
        x = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(s3)


        #Fifth block
        x = layers.Conv2D(512,3,activation="relu",padding="same")(x)
        x = layers.Conv2D(512,3,activation="relu",padding="same")(x)
        s4 = layers.Conv2D(512,3,activation="relu",padding="same")(x)
        x = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(s4)



        #1x1 convolutions for lateral connections
        s1 = layers.Conv2D(128,1,activation=None,padding="same")(s1)
        s2 = layers.Conv2D(256,1,activation=None,padding="same")(s2)
        s3 = layers.Conv2D(256,1,activation=None,padding="same")(s3)
        s4 = layers.Conv2D(256,1,activation=None,padding="same")(s4)


        #Decoder

        #First block
        x = layers.Conv2D(256,3,activation="relu",padding="same")(x)
        x = layers.UpSampling2D(size=(2,2),interpolation="bilinear")(x)
        x = layers.Conv2D(256,3,activation="relu",padding="same")(x)
        x = layers.add((x,s4))
        x = layers.Conv2D(256,3,activation="relu",padding="same")(x)

        #Second block
        x = layers.UpSampling2D(size=(2,2),interpolation="bilinear")(x)
        x = layers.Conv2D(256,3,activation="relu",padding="same")(x)
        x = layers.add((x,s3))
        x = layers.Conv2D(256,3,activation="relu",padding="same")(x)

        #Third block with intermediate output for count error
        x = layers.UpSampling2D(size=(2,2),interpolation="bilinear")(x)
        x = layers.Conv2D(256,3,activation="relu",padding="same")(x)
        x = layers.add((x,s2))
        x = layers.Conv2D(256,3,activation="relu",padding="same")(x)
        o1 = layers.Conv2D(1,1,activation=None,padding="same")(x)
        x = layers.Conv2D(128,3,activation="relu",padding="same")(x)

        #Fourth block
        x = layers.UpSampling2D(size=(2,2),interpolation="bilinear")(x)
        x = layers.Conv2D(128,3,activation="relu",padding="same")(x)
        x = layers.add((x,s1))
        x = layers.Conv2D(128,3,activation="relu",padding="same")(x)

        #Fifth block
        x = layers.UpSampling2D(size=(2,2),interpolation="bilinear")(x)
        x = layers.Conv2D(128,3,activation="relu",padding="same")(x)
        x = layers.Conv2D(128,3,activation="relu",padding="same")(x)
        o2 = layers.Conv2D(1,1,activation=None,padding="same")(x)

        #model = keras.Model(inputs=inputs, outputs=[o1,o2], name="mrcnet_model")
        model = keras.Model(inputs=inputs, outputs=o2, name="mrcnet_model")
        """
        print(x.shape)
        print(s1.shape)
        print(s2.shape)
        print(s3.shape)
        print(s4.shape)
        """
        print(model.summary())
        
        #Total params should be 20.3M according to Bahmanyar et al
        #keras.utils.plot_model(model,"mrc_graph.png")
        return model
    
    #Simplified model for testing
    def buildModel_simple(self):
        inputs = keras.Input(shape=(p.X,p.Y,1))
        
        x = layers.Conv2D(64,3,activation="relu",padding="same")(inputs)
        x = layers.Conv2D(64,3,activation="relu",padding="same")(x)
        x = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(x)
        x = layers.Conv2D(256,3,activation="relu",padding="same")(x)
        x = layers.UpSampling2D(size=(2,2),interpolation="bilinear")(x)
        x = layers.Conv2D(64,3,activation="relu",padding="same")(x)
        o2 = layers.Conv2D(1,1,activation="relu",padding="same")(x)
        
        model = keras.Model(inputs=inputs, outputs=o2, name="simple_model")
        print(model.summary())
        return model
        
        
        



        
        
