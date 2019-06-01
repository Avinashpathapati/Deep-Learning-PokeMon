# Train module
# Author: Andreas Pentaliotis
# Module to implement training of a model on the data.

import numpy as np
import random

from utility import load_images, randomize
from preprocessing import preprocess
from dcgan import DCGAN
from keras.preprocessing.image import ImageDataGenerator


fire = ["Charmander", "Charizard", "Charmeleon", "Flareon", "Growlithe", "Magmar", "Moltres", "Ninetales", "Ponyta", "Rapidash", "Vulpix"]
water = ["Blastoise", "Squirtle", "Wartortle", "Psyduck", "Golduck", "Polywag", "Polywhirl", "Seel", "Shellder", "Krabby", "Kingler",
         "Horsea", "Seadra", "Goldeen", "Seaking", "Staryu", "Magikarp", "Vaporeon"]
images = load_images("./mgan-data")

randomize(images)
images = preprocess(images)
  
dcgan = DCGAN(images.shape[1], images.shape[2], images.shape[3])
dcgan.summary()

data_generator = ImageDataGenerator(zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2, rotation_range=10)

dcgan.train(images, epochs=20000, batch_size=32, output_path="./output", save_interval=50, data_generator=data_generator)
