# Train module
# Author: Andreas Pentaliotis
# Module to implement training of a model on the data.

import numpy as np
import random

from utility import load_images, randomize
from preprocessing import preprocess
from augmentation import augment
from gan import GAN
from keras.images import ImageDataGenerator


images = load_images("./pokemon-data/Pikachu")

images = augment(images)
randomize(images)
images = preprocess(images)

gan = GAN(images.shape[1], images.shape[2], images.shape[3])
gan.summary()

gan.train(images, epochs=1000, batch_size=32, output_path="./output", save_interval=20, training_generator)
