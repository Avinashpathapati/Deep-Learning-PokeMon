# Test module
# Module to implement testing for other modules.



from utility import load_data
data = load_data("./pokemon-generation-one")
print(data["images"][0].shape)

from utility import plot
for image, label in zip(images, labels):
  plot(image, label)
