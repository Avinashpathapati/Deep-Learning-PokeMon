# Augmentation module
# Module to implement data augmentation.

import random
import numpy as np
import cv2 as cv


def scale(image):
  (height, width) = image.shape[:2] 
  scale_type = random.choice([cv.INTER_CUBIC, cv.INTER_AREA])
  height = int(1.2 * height) if scale_type == cv.INTER_AREA else int(height / 1.2)
  width = int(1.2 * width) if scale_type == cv.INTER_AREA else int(width / 1.2)
  return cv.resize(image, (width, height), interpolation = scale_type) 

def rotate(image):
  (rows, columns) = image.shape[:2]
  rotation_degree = random.choice([-10, -5, 5, 10])
  matrix = cv.getRotationMatrix2D((columns / 2, rows / 2), rotation_degree, 1) 
  return cv.warpAffine(image, matrix, (columns, rows), borderValue = (255, 255, 255)) 

def translate(image):
  (rows, columns) = image.shape[:2] 
  horizontal_shift = random.choice([-4, -3, -2, -1, 0, 1, 2, 3, 4])
  vertical_shift = random.choice([-4, -3, -2, -1, 0, 1, 2, 3, 4])
  matrix = np.float32([[1, 0, horizontal_shift], [0, 1, vertical_shift]]) 
  return cv.warpAffine(image, matrix, (columns, rows), borderValue = (255, 255, 255))

############# Removed zooming in / out to keep the generator parameters fixed. ##############
def augment(images, labels):
  print("augmenting images...")

  #scaled_images = [scale(x) for x in images]
  rotated_images = [rotate(x) for x in images]
  translated_images = [translate(x) for x in images]
  
  #images.extend(scaled_images)
  images.extend(rotated_images)
  images.extend(translated_images)
  labels = labels * 4

  return images, labels
