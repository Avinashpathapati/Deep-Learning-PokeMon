# Preprocessing module
# Author: Andreas Pentaliotis
# Module to implement preprocessing of the data.

import cv2 as cv
import numpy as np

from utility import normalize

def __resize(images, dimensions):
  images = [cv.resize(x, dimensions, interpolation=cv.INTER_CUBIC) for x in images]
  return images


def preprocess(images):
  print("preprocessing images...")

  images = __resize(images, dimensions=(128, 128))
  images = np.array(images)
  images = normalize(images, pixel_range=(-1, 1))

  return images
