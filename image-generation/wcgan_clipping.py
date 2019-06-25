"""An implementation of the improved WGAN described in https://arxiv.org/abs/1704.00028
The improved WGAN has a term in the loss function which penalizes the network if its
gradient norm moves away from 1. This is included because the Earth Mover (EM) distance
used in WGANs is only easy to calculate for 1-Lipschitz functions (i.e. functions where
the gradient norm has a constant upper bound of 1).
The original WGAN paper enforced this by clipping weights to very small values
[-0.01, 0.01]. However, this drastically reduced network capacity. Penalizing the
gradient norm is more natural, but this requires second-order gradients. These are not
supported for some tensorflow ops (particularly MaxPool and AveragePool) in the current
release (1.0.x), but they are supported in the current nightly builds
(1.1.0-rc1 and higher).
To avoid this, this model uses strided convolutions instead of Average/Maxpooling for
downsampling. If you wish to use pooling operations in your discriminator, please ensure
you update Tensorflow to 1.1.0-rc1 or higher. I haven't tested this with Theano at all.
The model saves images using pillow. If you don't have pillow, either install it or
remove the calls to generate_images.
"""
import argparse
import os
import numpy as np
from keras.models import Model, Sequential
from keras.optimizers import RMSprop
from keras.layers import Input, Dense, Reshape, Flatten,Lambda, concatenate
from keras.layers.merge import _Merge
from keras.layers.convolutional import Conv2DTranspose,Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Activation
from keras.optimizers import Adam
from keras.datasets import mnist
from keras import backend as K
from functools import partial
from utility import generate_images, save
import matplotlib.pyplot as plt

try:
    from PIL import Image
except ImportError:
    print('This script depends on pillow! '
          'Please install it (e.g. with pip install pillow)')
    exit()

batch_size = 64

# The training ratio is the number of discriminator updates
# per generator update. The paper uses 5.

class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this
    outputs a random point on the line between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could
    think of. Improvements appreciated."""
    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


class GAN():

    def __init__(self, height, width, depth):
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.0002)

        # Build and compile the critic
        self.critic = self.make_critic()
        self.critic.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.make_generator()

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.critic.trainable = False

        # The critic takes generated images as input and determines validity
        valid = self.critic(img)

        # The combined model  (stacked generator and critic)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)


    def make_generator(self):
        model = Sequential()

        model.add(Dense(4 * 4 * 512, input_dim=100))
        model.add(Reshape((4, 4, 512)))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        
        model.add(Conv2DTranspose(256, kernel_size=5, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        model.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        model.add(Conv2DTranspose(64, kernel_size=5, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        model.add(Conv2DTranspose(32, kernel_size=5, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        model.add(Conv2DTranspose(self.channels, kernel_size=5, strides=2, padding="same"))
        model.add(Activation("tanh"))
        model.summary()
        
        return model


    def make_critic(self):

        input_shape = (self.img_rows, self.img_cols, self.channels)
        
        # Build the model
        model = Sequential()
      
        model.add(Conv2D(64, kernel_size=5, strides=2, padding="same", input_shape=input_shape))
        #model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(128, kernel_size=5, strides=2, padding="same"))
        #model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(256, kernel_size=5, strides=2, padding="same"))
        #model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(512, kernel_size=5, strides=2, padding="same"))
        #model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Flatten())
        model.add(Dense(1, kernel_initializer='he_normal'))
        model.summary()

        return model

    def train(self,images, epochs, output_path, save_interval, data_generator):

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))
        g_loss_arr = []
        d_loss_arr = []

        for epoch in range(epochs):
            print("Running epoch {}/{}...".format(epoch, epochs))
            for _ in range(self.n_critic):
                print(_)
                training_generator = data_generator.flow(images, batch_size=int(batch_size))
                imgs = training_generator.next()
                
                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Generate a batch of new images
                gen_imgs = self.generator.predict(noise)

                # Train the critic
                d_loss_real = self.critic.train_on_batch(imgs, valid)
                d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                # Clip critic weights
                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)


            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))
            d_loss_arr.append(1 - d_loss[0])
            g_loss_arr.append(1 - g_loss[0])

            # If at save interval => save generated image samples
            if epoch % 50 == 0:

                self.generator.save(output_path + "/generator.h5")
                self.critic.save(output_path + "/discriminator.h5")
                images_gen = generate_images(self.generator, 10)
                save(images_gen, output_path + "/epoch-" + str(epoch))
                print("saving training history...")
                plt.plot([x for x in d_loss_arr])
                plt.title("Discriminator training loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.savefig(output_path + "/epoch-" + str(epoch) + "/discriminator-training-loss")
                plt.close()

                plt.plot([x for x in g_loss_arr])
                plt.title("Generator training loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.savefig(output_path + "/epoch-" + str(epoch) + "/generator-training-loss")
                plt.close()
