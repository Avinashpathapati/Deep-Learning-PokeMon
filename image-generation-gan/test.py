# Test module
# Module to implement testing for other modules. - To be removed.


"""
from utility import load_data
images, labels = load_data("./pokemon-generation-one")
print(data["images"][0].shape)

from utility import plot
for image, label in zip(images, labels):
  plot(image, label)
"""

#from gan import GAN
#m = GAN(256, 256, 3, 10)
#m.summary()

x = [[1, 10], [2, 3]]
y = [[2,3], [2,2]]
import numpy as np
print(np.average(x, axis=0))
print([x[0] for x in x])