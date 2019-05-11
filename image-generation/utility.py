# Utility module
# Author: Andreas Pentaliotis
# Module to implement utility functions.

import os
import cv2 as cv
import random


def plot(image, name):
  cv.imshow(name, image)
  cv.waitKey(0)
  cv.destroyWindow(name)

def read_image(path):
  return cv.imread(path, cv.IMREAD_COLOR)

def save(image, path, name):
  cv.imwrite(os.path.join(path, name), image)

def load_data(path):
  print("loading images...")
  
  images = []
  for directory in os.listdir(path + "/"):
    for filename in os.listdir(path + "/" + str(directory) + "/"):
      image = read_image(path + "/" + str(directory) + "/" + str(filename))
      images.append(image)
  
  return images

def randomize(images):
  print("shuffling images...")
  random.Random(1).shuffle(images)