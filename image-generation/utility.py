# Utility module
# Author: Andreas Pentaliotis
# Module to implement utility functions.

import os
import cv2 as cv
import random
import argparse
import numpy as np


def plot(image, name):
  cv.imshow(name, image)
  cv.waitKey(0)
  cv.destroyWindow(name)

def read_image(path):
  return cv.imread(path, cv.IMREAD_COLOR)

def load_images(path):
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

def parse_input_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument("-n", "--number", required=True, help="number of images to generate")
  arguments = vars(parser.parse_args())
  
  if int(arguments["number"]) <= 0:
    raise ValueError("number of images should be positive")

  return arguments

def generate_images(generator, number):
  print("generating images...")

  noise = np.random.normal(0, 1, (number, 100))
  images = generator.predict(noise)
  
  return images

def save(images):
  print("saving generated images to ./output...")
  
  if not os.path.isdir("./output"):
    os.mkdir("./output")
  for image in images:
    cv.imwrite(os.path.join("./output", str(images.index(image)) + ".jpg"), image)
