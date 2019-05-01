# Utility module
# Author: Andreas Pentaliotis
# Module to implement utility functions for generator.

import matplotlib.pyplot as plt
import argparse
import os
import cv2 as cv
import pandas as pd


def plot(image, name):
  cv.imshow(name, image)
  cv.waitKey(0)
  cv.destroyWindow(name)

def read_image(path):
  return cv.imread(path, cv.IMREAD_UNCHANGED)

def resize(image):
  return cv.resize(image, (256, 256))

def load_data(path):
  print("loading images...")
  
  # Load the images and labels into a dataframe.
  data = pd.DataFrame()
  images = []
  labels = []
  for directory in os.listdir(path + "/"):
    for filename in os.listdir(path + "/" + str(directory) + "/"):
      if str(directory) == "Abra":
        image = read_image(path + "/" + str(directory) + "/" + str(filename))
        image = resize(image)
        images.append(image)
        labels.append(str(directory))

  data["images"] = images
  data["labels"] = labels
  
  return data
