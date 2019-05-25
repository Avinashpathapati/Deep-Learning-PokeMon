# Train module
# Author: Andreas Pentaliotis
# Module to implement training of a model on the data.

import numpy as np
import random

from utility import load_images, randomize
from preprocessing import preprocess
from gan import GAN
from keras.preprocessing.image import ImageDataGenerator


fire = ["Charmander", "Charizard", "Charmeleon", "Flareon", "Growlithe", "Magmar", "Moltres", "Ninetales", "Ponyta", "Rapidash", "Vulpix"]
water = ["Blastoise", "Squirtle", "Wartortle", "Psyduck", "Golduck", "Polywag", "Polywhirl", "Seel", "Shellder", "Krabby", "Kingler",
         "Horsea", "Seadra", "Goldeen", "Seaking", "Staryu", "Magikarp", "Vaporeon"]
images = load_images("./pokemon-data-cleaned", "Mew")

randomize(images)
images = preprocess(images)

gan = GAN(images.shape[1], images.shape[2], images.shape[3])
gan.summary()

data_generator = ImageDataGenerator(zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2, rotation_range=10)

gan.train(images, epochs=5000, batch_size=32, output_path="./output", save_interval=20, data_generator=data_generator)
