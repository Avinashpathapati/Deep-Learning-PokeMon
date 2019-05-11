# Preprocessing module
# Author: Andreas Pentaliotis
# Module to implement preprocessing of the data.

import cv2 as cv
import numpy as np
from random import shuffle


def __resize(images):
  images = [cv.resize(x, (64, 64)) for x in images]
  return images

def preprocess(images):
  print("preprocessing data...")

  images = __resize(images)
  shuffle(images)
  images = np.array(images, dtype=np.uint8)
  images = np.divide(images, 255)
  
  return images
