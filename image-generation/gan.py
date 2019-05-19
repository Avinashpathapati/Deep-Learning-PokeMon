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
    self.adversarial.compile(loss="binary_crossentropy", optimizer=optimizer)

  def __build_discriminator(self):
    input_shape = (self.height, self.width, self.depth)
    
    # Build the model
    model = Sequential()
  
    model.add(Conv2D(128, kernel_size=5, strides=2, padding="same", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(256, kernel_size=5, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(512, kernel_size=5, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(1024, kernel_size=5, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))

    return model

  def __build_generator(self):
    # Determine initial dimensions.
    height = int(self.height / 16)
    width = int(self.width / 16)

    # Build the model.
    model = Sequential()

    model.add(Dense(height * width * 1024, input_dim=100))
    model.add(Reshape((height, width, 1024)))
    
    model.add(Conv2DTranspose(512, kernel_size=5, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv2DTranspose(256, kernel_size=5, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding="same"))
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

  def train(self, images, epochs, batch_size, output_path, save_interval, data_generator):
    if batch_size > images.shape[0]:
      raise ValueError("batch size should be less than size of data")

    if not isinstance(save_interval, int):
      raise ValueError("save interval should be an integer")

    batches = int(images.shape[0] / batch_size)
    training_generator = data_generator.flow(images, batch_size=int(batch_size / 2))
    
    discriminator_history_real = []
    discriminator_history_fake = []
    generator_history = []
    for epoch in range(1, epochs + 1):
      discriminator_statistics_real = []
      discriminator_statistics_fake = []
      generator_statistics = []
      for _ in range(batches):
        # Select a mini batch of real images randomly, with size half of batch size. Account for the
        # case where the size of images is not divisible by batch size.
        real_images = training_generator.next()
        if real_images.shape[0] != int(batch_size / 2):
          real_images = training_generator.next()
        real_labels = np.ones((int(batch_size / 2), 1))

        # Generate fake images from noise, with size half of batch size.
        noise = np.random.normal(0, 1, (int(batch_size / 2), 100))
        fake_images = self.generator.predict(noise)
        fake_labels = np.zeros((int(batch_size / 2), 1))

        # Train the discriminator.
        discriminator_statistics_real.append(self.discriminator.train_on_batch(real_images, real_labels))
        discriminator_statistics_fake.append(self.discriminator.train_on_batch(fake_images, fake_labels))

        # Sample data points from the noise distribution, with size of batch size and create
        # real labels for them.
        noise = np.random.normal(0, 1, (batch_size, 100))
        real_labels = np.ones((batch_size, 1))

        # Train the generator.
        generator_statistics.append(self.adversarial.train_on_batch(noise, real_labels))

      discriminator_history_real.append(np.average(discriminator_statistics_real, axis=0))
      discriminator_history_fake.append(np.average(discriminator_statistics_fake, axis=0))
      generator_history.append(np.average(generator_statistics, axis=0))

      # Print the statistics for the current epoch.
      print()
      print("Epoch %d/%d" % (epoch, epochs))
      print("--------------------")
      print("Discriminator: [loss real: %f, acc real: %.2f%%, loss fake: %f, acc fake: %.2f%%]"
             % (discriminator_history_real[-1][0], 100 * discriminator_history_real[-1][1],
             discriminator_history_fake[-1][0], 100 * discriminator_history_fake[-1][1]))
      print("Generator: [loss: %f]" % generator_history[-1])

      if epoch % save_interval == 0:
        if not os.path.isdir(str(output_path)):
          os.mkdir(str(output_path))

        # Save the generator, the discriminator and a sample of fake images.
        self.save_models(output_path + "/epoch-" + str(epoch))
        images = generate_images(self.generator, 10)
        save(images, output_path + "/epoch-" + str(epoch))

        # Save the training history up to the current epoch.
        print("saving training history...")

        plt.plot([x[1] for x in discriminator_history_real])
        plt.plot([x[1] for x in discriminator_history_fake])
        plt.title("Discriminator training accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(["Real", "Fake"], loc="upper left")
        plt.savefig(output_path + "/epoch-" + str(epoch) + "/discriminator-training-accuracy")
        plt.close()

        plt.plot([x[0] for x in discriminator_history_real])
        plt.plot([x[0] for x in discriminator_history_fake])
        plt.title("Discriminator training loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(["Real", "Fake"], loc="upper left")
        plt.savefig(output_path + "/epoch-" + str(epoch) + "/discriminator-training-loss")
        plt.close()

        plt.plot([x for x in generator_history])
        plt.title("Generator training loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(output_path + "/epoch-" + str(epoch) + "/generator-training-loss")
        plt.close()

  def save_models(self, path):
    print("saving models...")
    if not os.path.isdir(str(path)):
      os.mkdir(str(path))

    self.generator.save(str(path) + "/generator.h5")
    self.discriminator.save(str(path) + "/discriminator.h5")
