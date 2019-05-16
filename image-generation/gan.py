# GAN module
# Author: Andreas Pentaliotis
# Module to implement a generative adversarial network.

from keras.models import Sequential
from keras.models import Model
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.core import Activation
from keras.layers import LeakyReLU
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers.core import Reshape
from keras.optimizers import Adam
from keras import backend as K
K.set_image_dim_ordering("tf")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from utility import generate_images, save


class GAN():
  def __init__(self, height, width, depth):
    self.height = height
    self.width = width
    self.depth = depth
    self.__build()

  def __build(self):
    optimizer = Adam(lr=0.0002, beta_1=0.5)

    # Build the generator and discriminator and compile the discriminator.
    self.generator = self.__build_generator()
    self.discriminator = self.__build_discriminator()
    self.discriminator.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    # Build and compile the adversarial network.
    self.discriminator.trainable = False
    noise = Input(shape=(100,))
    fake_image = self.generator(noise)
    label = self.discriminator(fake_image)
    self.adversarial = Model(noise, label)
    self.adversarial.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

  def __build_discriminator(self):
    input_shape = (self.height, self.width, self.depth)
    
    # Build the model
    model = Sequential()
  
    model.add(Conv2D(64, kernel_size=5, strides=2, padding="same", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(128, kernel_size=5, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
  
    model.add(Conv2D(256, kernel_size=5, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))

    return model

  def __build_generator(self):
    # Determine initial dimensions.
    height = int(self.height / 8)
    width = int(self.width / 8)

    # Build the model.
    model = Sequential()

    model.add(Dense(height * width * 256, input_dim=100))
    model.add(Reshape((height, width, 256)))
    
    model.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    
    model.add(Conv2DTranspose(64, kernel_size=5, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv2DTranspose(self.depth, kernel_size=5, strides=2, padding="same"))
    model.add(Activation("tanh"))

    return model

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

  def train(self, images, epochs, batch_size, output_path, save_interval):
    if batch_size > images.shape[0]:
      raise ValueError("batch size should be less than size of data")

    if not isinstance(save_interval, int):
      raise ValueError("save interval should be an integer")

    if not os.path.isdir(str(output_path)):
      os.mkdir(str(output_path))

    batches = int(images.shape[0] / batch_size)
    discriminator_history = []
    adversarial_history = []
    for epoch in range(1, epochs + 1):
      discriminator_statistics = []
      adversarial_statistics = []
      for _ in range(batches):
        # Select a mini batch of real images randomly, with size half of batch size. 
        indices = np.random.randint(0, images.shape[0], int(batch_size / 2))
        real_images = images[indices]
        real_labels = np.ones((int(batch_size / 2), 1))
      
        # Generate fake images from noise, with size half of batch size.
        noise = np.random.normal(0, 1, (int(batch_size / 2), 100))
        fake_images = self.generator.predict(noise)
        fake_labels = np.zeros((int(batch_size / 2), 1))

        # Train the discriminator.
        discriminator_statistics_real = self.discriminator.train_on_batch(real_images, real_labels)
        discriminator_statistics_fake = self.discriminator.train_on_batch(fake_images, fake_labels)
        discriminator_statistics.append(0.5 * np.add(discriminator_statistics_real, discriminator_statistics_fake))

        # Sample data points from the noise distribution, with size of batch size and create
        # real labels for them.
        noise = np.random.normal(0, 1, (batch_size, 100))
        real_labels = np.ones((batch_size, 1))

        # Train the generator.
        adversarial_statistics.append(self.adversarial.train_on_batch(noise, real_labels))

      discriminator_history.append(np.average(discriminator_statistics, axis=0))
      adversarial_history.append(np.average(adversarial_statistics, axis=0))

      print("Epoch %d/%d [Discriminator]: [loss: %f, acc.: %.2f%%] [Adversarial loss: %f, acc: %.2f%%]"
             % (epoch, epochs, discriminator_history[-1][0], 100 * discriminator_history[-1][1],
             adversarial_history[-1][0], 100 * adversarial_history[-1][1]))

      if epoch % save_interval == 0:
        # Save the generator and a sample of fake images.
        self.save_generator(output_path + "/epoch-" + str(epoch))
        images = generate_images(self.generator, 10)
        save(images, output_path + "/epoch-" + str(epoch))

        # Save the training history up to the current epoch.
        plt.plot([x[1] for x in discriminator_history])
        plt.plot([x[0] for x in discriminator_history])
        plt.title("Discriminator training")
        plt.xlabel("Epoch")
        plt.legend(["Accuracy", "Loss"], loc="upper left")
        plt.savefig(output_path + "/epoch-" + str(epoch) + "/discriminator-training")
        plt.close()

        plt.plot([x[1] for x in adversarial_history])
        plt.plot([x[0] for x in adversarial_history])
        plt.title("Adversarial training")
        plt.xlabel("Epoch")
        plt.legend(["Accuracy", "Loss"], loc="upper left")
        plt.savefig(output_path + "/epoch-" + str(epoch) + "/adversarial-training")
        plt.close()

  def save_generator(self, path):
    if not os.path.isdir(str(path)):
      os.mkdir(str(path))

    self.generator.save(str(path) + "/generator.h5")
