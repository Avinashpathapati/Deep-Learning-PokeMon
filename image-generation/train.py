# Train module
# Author: Andreas Pentaliotis
# Module to implement training of a model on the data.

import numpy as np
import random

from utility import load_data, randomize
from preprocessing import preprocess
from augmentation import augment
from gan import GAN


images = load_data("./pokemon-generation-one")

images = augment(images)
randomize(images)
images = preprocess(images)

gan = GAN(images.shape[1], images.shape[2], images.shape[3])
gan.summary()

gan.train(images, epochs=2, batch_size=32)
gan.save()
