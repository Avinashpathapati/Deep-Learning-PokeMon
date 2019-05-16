# Utility module
# Author: Andreas Pentaliotis
# Module to implement utility functions.

import os
import cv2 as cv
import random
import argparse
import numpy as np
from scipy.misc import imsave


def plot(image, name):
  cv.imshow(name, image)
  cv.waitKey(0)
  cv.destroyWindow(name)

def read_image(path):
  image = cv.imread(path, cv.IMREAD_COLOR)
  return image

def load_images(path):
  print("loading images...")

  images = []
  for filename in os.listdir(path + "/"):
    image = read_image(path + "/" + str(filename))
    images.append(image)
  
  return images

def randomize(images):
  print("shuffling images...")
  random.Random(1).shuffle(images)

def parse_input_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument("-n", "--number", required=True, help="number of images to generate")
  arguments = vars(parser.parse_args())
  
  if int(arguments["number"]) <= 0:
    raise ValueError("number of images should be positive")

  return arguments

def normalize(images, pixel_range):
  if pixel_range == (-1, 1):
    # Normalize the pixel values in the images to [-1, 1]
    images = 2.0 * (images - np.min(images)) / np.ptp(images) - 1
  elif pixel_range == (0, 1):
    # Normalize the pixel values in the images to [0, 1]
    images = (images - np.min(images)) / np.ptp(images)
  else:
    raise ValueError("invalid pixel range")
  
  return images

def generate_images(generator, number):
  print("generating images...")

  noise = np.random.normal(0, 1, (number, 100))
  images = generator.predict(noise)
  images = normalize(images, pixel_range=(0, 1))

  return images

def save(images, path):
  print("saving generated images to " + str(path) + "...")
  
  if not os.path.isdir(str(path)):
    os.mkdir(str(path))

  image_name = 0  
  for image in images:
    imsave(os.path.join(str(path), str(image_name) + ".jpg"), image)
    image_name += 1
