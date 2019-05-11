# Augmentation module
# Author: Andreas Pentaliotis
# Module to implement data augmentation.

import random
import numpy as np
import cv2 as cv


def __rotate(images):
  (rows, columns) = images[0].shape[:2]
  rotation_degree = random.choice([-10, -5, 5, 10])
  matrix = cv.getRotationMatrix2D((columns / 2, rows / 2), rotation_degree, 1)
  images = [cv.warpAffine(x, matrix, (columns, rows), borderValue = (255, 255, 255)) for x in images]
  return images

def __translate(images):
  (rows, columns) = images[0].shape[:2] 
  horizontal_shift = random.choice([-2, -1, 0, 1, 2])
  vertical_shift = random.choice([-2, -1, 0, 1, 2])
  matrix = np.float32([[1, 0, horizontal_shift], [0, 1, vertical_shift]]) 
  images = [cv.warpAffine(x, matrix, (columns, rows), borderValue = (255, 255, 255)) for x in images]
  return images

def augment(images):
  print("augmenting images...")

  rotated_images = __rotate(images)
  translated_images = __translate(images)
  
  images.extend(rotated_images)
  images.extend(translated_images)

  return images
