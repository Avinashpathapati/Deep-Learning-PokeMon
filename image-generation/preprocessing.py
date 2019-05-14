# Preprocessing module
# Author: Andreas Pentaliotis
# Module to implement preprocessing of the data.

import cv2 as cv
import numpy as np


def __resize(images):
  images = [cv.resize(x, (256, 256)) for x in images]
  return images

def preprocess(images):
  print("preprocessing images...")

  images = __resize(images)
  images = np.array(images)
  images = np.divide(images, 255)
  
  return images
