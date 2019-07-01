import argparse
import os
import numpy as np
import pickle
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
K.set_learning_phase(1)
import tensorflow as tf
from functools import partial
from utility import generate_images, save
import matplotlib.pyplot as plt
from utils import *

try:
    from PIL import Image
except ImportError:
    print('This script depends on pillow! '
          'Please install it (e.g. with pip install pillow)')
    exit()

batch_size = 64
version = 'newPokemon'
newPoke_path = './' + version



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
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(128, kernel_size=5, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(256, kernel_size=5, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(512, kernel_size=5, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Flatten())
        model.add(Dense(1, kernel_initializer='he_normal'))
        model.summary()

        return model

    def train(self,images, epochs, output_path, save_interval, data_generator):

  
        g_loss_arr = []
        d_loss_arr = []

        batches = int(images.shape[0] / batch_size)

         # Build and compile the critic
        self.critic = self.make_critic()
        self.generator = self.make_generator()



        netD_real_input = Input(shape=(self.img_rows, self.img_cols, self.channels))
        noisev = Input(shape=(self.latent_dim,))

        netD_fake_input = self.generator(noisev)


        loss_real = K.mean(self.critic(netD_real_input))
        loss_fake = K.mean(self.critic(netD_fake_input))

        #wgan discriminator loss
        loss = loss_fake - loss_real


        training_updates = RMSprop(lr=0.0002).get_updates(self.critic.trainable_weights,[],loss)
        netD_train = K.function([netD_real_input, noisev],
                                [loss_real, loss_fake],
                                training_updates)

        #wgan generator loss
        loss = -loss_fake 
        training_updates = RMSprop(lr=0.0002).get_updates(self.generator.trainable_weights,[], loss)
        netG_train = K.function([noisev], [loss], training_updates)



        for epoch in range(epochs):
            print("Running epoch {}/{}...".format(epoch, epochs))
            for j in range(batches):
                for _ in range(self.n_critic):
                    print(_)

                    training_generator = data_generator.flow(images, batch_size=int(batch_size))
                    imgs = training_generator.next()
                    noise = np.random.normal(0, 1, (batch_size, 100))

                    errD_real, errD_fake  = netD_train([imgs, noise])
                    errD = errD_real - errD_fake

                    # Clip critic weights
                    for l in self.critic.layers:
                        weights = l.get_weights()
                        weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                        l.set_weights(weights)


                noise = np.random.normal(0, 1, (batch_size, 100))
                errG, = netG_train([noise])
                print('train:[%d],d_loss:%f,g_loss:%f' % (epoch, errD, errG))
            d_loss_arr.append(errD)
            g_loss_arr.append(errG)



            # If at save interval => save generated image samples
            if epoch % 50 == 0:

                self.generator.save(output_path + "/generator.h5")
                self.critic.save(output_path + "/discriminator.h5")
                images_gen = generate_images(self.generator, 7)

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

                #saving into pickle file

                with open(output_path + "/epoch-" + str(epoch)+'/trainHistoryDict_'+"epoch_"+str(epoch)+"_disc", 'wb') as file_pi:
                    pickle.dump(d_loss_arr, file_pi)

                with open(output_path + "/epoch-" + str(epoch)+'/trainHistoryDict_'+"epoch_"+str(epoch)+"_gen", 'wb') as file_pi:
                    pickle.dump(g_loss_arr, file_pi)
