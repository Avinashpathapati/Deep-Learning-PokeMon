# Train module
# Author: Andreas Pentaliotis
# Module to implement training of a model on the data.

import numpy as np

from utility import load_data
from preprocessing import preprocess
from augmentation import augment
from gan import GAN


images = load_data("./pokemon-generation-one")

images = augment(images)
images = preprocess(images)

gan = GAN(images.shape[1], images.shape[2], images.shape[3])
gan.summary()

gan.train(images, epochs=500, batch_size=32)
gan.save()
