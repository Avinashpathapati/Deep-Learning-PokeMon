# GAN module
# Module to implement a generative adversarial network.

from keras.models import Sequential
from keras.models import Model
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.constraints import maxnorm
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers.core import Reshape
from keras.layers import UpSampling2D
from keras import backend as K
K.set_image_dim_ordering("tf")


class GAN():
  def __init__(self, height, width, depth):
    self.height = height
    self.width = width
    self.depth = depth
    self.__build()

  def __build(self):
    # Build the generator and discriminator and compile the discriminator.
    self.generator = self.__build_generator()
    self.discriminator = self.__build_discriminator()
    self.discriminator.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Build and compile the adversarial network.
    self.discriminator.trainable = False
    input_noise = Input(shape=(100,))
    fake_image = self.generator(input_noise)
    decision = self.discriminator(fake_image)
    self.adversarial = Model(input_noise, decision)
    self.adversarial.compile(loss="binary_crossentropy", optimizer="adam")

  def __build_discriminator(self):
    input_shape = (self.height, self.width, self.depth)
    
    # Build the model
    model = Sequential()
  
    model.add(Conv2D(64, (5, 5), input_shape=input_shape, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
  
    model.add(Conv2D(128, (5, 5), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (5, 5), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
  
    model.add(Conv2D(512, (5, 5), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation="sigmoid"))

    return model

  def __build_generator(self):
    # Determine initial dimensions.
    height = int(self.height / 32)
    width = int(self.width / 32)

    # Build the model.
    model = Sequential()

    model.add(Dense(height * width * 512, input_dim=100))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Reshape((height, width, 512)))

    model.add(UpSampling2D())
    model.add(Conv2D(256, (5, 5), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(128, (5, 5), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(64, (5, 5), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(32, (5, 5), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(16, (5, 5), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv2D(self.depth, (5, 5), padding="same"))
    model.add(Activation("sigmoid"))

    return model
