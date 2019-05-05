# GAN module
# Author: Andreas Pentaliotis
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
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class GAN():
  def __init__(self, height, width, depth, classes):
    self.height = height
    self.width = width
    self.depth = depth
    self.classes = classes
    self.__build()

  def summary(self):
    print()
    print()
    print("ADVERSARIAL")
    print("--------------------")
    self.adversarial.summary()
    print()
    print("DISCRIMINATOR")
    print("--------------------")
    self.discriminator.summary()
    print()
    print("GENERATOR")
    print("--------------------")
    self.generator.summary()

  def train(self, images, labels, epochs, batch_size):
    for epoch in range(epochs):
      # Select a mini batch of images randomly.
      indices = np.random.randint(0, images.shape[0], batch_size)
      real_images = images[indices]
      real_labels = labels[indices]
      
      # Generate fake images from noise.
      noise = np.random.normal(0, 1, (batch_size, 100))
      fake_images = self.generator.predict(noise)

      # Train the discriminator with supervised learning on real images and get the
      # predictions of the discriminator 
      discriminator_loss = self.discriminator.train_on_batch(real_images, real_labels)
      discriminator_predictions = 
    pass

  def __build(self):
    # Build the generator and discriminator and compile the discriminator.
    self.generator = self.__build_generator()
    self.discriminator = self.__build_discriminator()
    self.discriminator.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Build and compile the adversarial network.
    self.discriminator.trainable = False
    noise = Input(shape=(100,))
    fake_image = self.generator(noise)
    label = self.discriminator(fake_image)
    self.adversarial = Model(noise, label)
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

    model.add(Dense(self.classes, activation="softmax"))

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
