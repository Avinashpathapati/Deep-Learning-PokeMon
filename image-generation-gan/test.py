# Test module
# Module to implement testing for other modules.


"""
from utility import load_data
images, labels = load_data("./pokemon-generation-one")
print(data["images"][0].shape)

from utility import plot
for image, label in zip(images, labels):
  plot(image, label)
"""

from gan import GAN
m = GAN(256, 256, 3)
m.adversarial.summary()
m.discriminator.summary()
m.generator.summary()