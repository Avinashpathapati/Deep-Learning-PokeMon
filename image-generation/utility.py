# Utility module
# Author: Andreas Pentaliotis
# Module to implement utility functions.

import os
import cv2 as cv
import random
import argparse
import numpy as np
import scipy.misc


def plot(image, name):
  cv.namedWindow(name, cv.WINDOW_NORMAL)
  cv.imshow(name, image)
  cv.waitKey(0)
  cv.destroyWindow(name)


def read_image(path):
  image = cv.imread(path, cv.IMREAD_COLOR)
  return image


def load_images(path, pokemon=None):
  print("loading images...")

  images = []
  for directory in os.listdir(path + "/"):
    if "store" not in directory.lower():
      for filename in os.listdir(path + "/" + directory + "/"):
        if not "store" in filename.lower():
          if pokemon is not None and directory in pokemon or pokemon is None:
            image = read_image(path + "/" + directory + "/" + filename)
            images.append(image)
  
  return images


def randomize(images):
  print("shuffling images...")
  random.Random(1).shuffle(images)


def parse_input_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument("-n", "--number", required=True, help="number of images to generate")
  arguments = vars(parser.parse_args())
  
  if not arguments["number"].isdigit() or int(arguments["number"]) == 0:
    raise ValueError("number of images should be a positive integer")

  return arguments


def normalize(images, pixel_range):
  images = (images - np.min(images)) / np.ptp(images) * (pixel_range[1] - pixel_range[0]) + pixel_range[0]

  return images


def generate_images(generator, number):
  print("generating images...")
  noise = np.random.uniform(-1.0, 1.0, size=[number, 100]).astype(np.float32)
  images = generator.predict(noise)
  return images


def save(images, path):
  print("saving images to " + str(path) + "...")
  
  if not os.path.isdir(str(path)):
    os.mkdir(str(path))

  image_name = 0  
  for image in images:
    scipy.misc.imsave(os.path.join(str(path), str(image_name) + ".jpg"), image)
    #cv.imwrite(os.path.join(str(path), str(image_name) + ".jpg"), image)
    image_name += 1
