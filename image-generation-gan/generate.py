# Generator module
# Author: Andreas Pentaliotis
# Module to implement image generation on the given images.

from utility import plot, load_data


data = load_data("./pokemon-generation-one")

for image, label in zip(data["images"], data["labels"]):
  plot(image, label)
