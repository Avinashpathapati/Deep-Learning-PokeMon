# GAN module
# Module to implement a generative adversarial network.

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.constraints import maxnorm
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers.core import BatchNormalization
from keras.layers.core import Reshape
from keras.layers.core import UpSampling2D
from keras import backend as K


def build_discriminator(height, width, depth):
  # Determine the input shape.
  if K.image_data_format() == "channels_first":
    input_shape = (depth, height, width)
  else:
    input_shape = (height, width, depth)

  # Build the model
  model = Sequential()
  
  model.add(Conv2D(128, (11, 11), input_shape=input_shape, padding="same"))
  model.add(BatchNormalization())
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  
  model.add(Conv2D(256, (7, 7), padding="same"))
  model.add(BatchNormalization())
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(512, (5, 5), padding="same"))
  model.add(BatchNormalization())
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  
  model.add(Conv2D(1024, (3, 3), padding="same"))
  model.add(BatchNormalization())
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Flatten())
  model.add(Dense(2028))
  model.add(BatchNormalization())
  model.add(Activation("relu"))
  model.add(Dropout(0.5))

  model.add(Dense(1, activation="sigmoid"))

  return model

def build_generator(height, width, depth):
  # Build the model.
  model = Sequential()

  model.add(Dense(1024, input_shape=100, kernel_constraint=maxnorm(3)))
  model.add(BatchNormalization())
  model.add(Activation("relu"))
  model.add(Reshape((int(height / 16), int(width / 16), 256)))

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

  model.add(Conv2D(depth, (5, 5), padding="same"))
  model.add(Activation("sigmoid"))

  return model
