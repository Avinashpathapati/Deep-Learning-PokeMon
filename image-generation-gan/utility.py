# Utility module
# Module to implement utility functions for generator.

import os
import cv2 as cv


def plot(image, name):
  cv.imshow(name, image)
  cv.waitKey(0)
  cv.destroyWindow(name)

def read_image(path):
  return cv.imread(path, cv.IMREAD_COLOR)

def resize(image):
  return cv.resize(image, (25, 25))

def load_data(path):
  print("loading images...")
  
  # Load the images and labels, while resizing them.
  images = []
  labels = []
  for directory in os.listdir(path + "/"):
    for filename in os.listdir(path + "/" + str(directory) + "/"):
        image = read_image(path + "/" + str(directory) + "/" + str(filename))
        image = resize(image)
        images.append(image)
        labels.append(str(directory))
  
  return images, labels
