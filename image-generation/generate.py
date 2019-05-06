# Generate module
# Author: Andreas Pentaliotis
# Module to implement image generation.

from keras.models import load_model
import numpy as np
import argparse
import os

from utility import save


# Parse the input arguments.
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--number", required=True, help="number of images to generate")
arguments = vars(parser.parse_args())

if int(arguments["number"]) <= 0:
  raise ValueError("number of images should be positive")

# Load the model and generate the images.
print("generating images...")
generator = load_model("generator.h5")
noise = np.random.normal(0, 1, (arguments["number"], 100))
images = generator.predict(noise)

# Save the images.
print("saving images to ./output...")
if not os.path.isdir("./output"):
  os.mkdir("./output")
for image in images:
  save(image, "./output", str(images.index(image)) + ".jpg")
