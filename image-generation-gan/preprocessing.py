# Preprocessing module
# Author: Andreas Pentaliotis
# Module to implement preprocessing of the data.

import cv2 as cv
import numpy as np
from random import shuffle
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

  
def randomize(images, labels):
  data = list(zip(images, labels))
  shuffle(data)
  images[:], labels[:] = zip(*data)
  return images, labels

def resize(images):
  images = [cv.resize(x, (256, 256)) for x in images]
  return images

def preprocess(images, labels):
  print("preprocessing data...")

  images = resize(images)
  images, labels = randomize(images, labels)
  
  # One hot encode the labels.
  binarizer = LabelBinarizer()
  labels = binarizer.fit_transform(labels)

  images = np.array(images, dtype=np.uint8)
  images = np.divide(images, 255)
  
  return images, labels
