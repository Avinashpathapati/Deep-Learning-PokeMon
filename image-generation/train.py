# Train module
# Author: Andreas Pentaliotis
# Module to implement training of a model on the data.

import numpy as np
import random

from utility import load_images, randomize
from preprocessing import preprocess
from gan import GAN
from keras.preprocessing.image import ImageDataGenerator


images = load_images("./pokemon-data/Pikachu")

randomize(images)
images = preprocess(images)

gan = GAN(images.shape[1], images.shape[2], images.shape[3])
gan.summary()

data_generator = ImageDataGenerator(zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1, rotation_range=5)

gan.train(images, epochs=1000, batch_size=32, output_path="./output", save_interval=20, data_generator=data_generator)
