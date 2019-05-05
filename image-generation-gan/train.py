# Train module
# Author: Andreas Pentaliotis
# Module to implement training of a model on the data.

import numpy as np

from utility import load_data
from preprocessing import preprocess
from augmentation import augment
from gan import GAN


images, labels = load_data("./pokemon-generation-one")
classes = len(np.unique(labels))

images, labels = augment(images, labels)
images, labels = preprocess(images, labels)

gan = GAN(images.shape[1], images.shape[2], 3, classes)
gan.summary()

gan.train(images, labels, epochs=1, batch_size=5)

"""
history = model.fit(x_train, y_train, validation_split=0.25, epochs=1, batch_size=32)
model.save("model.h5")

plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.title("Model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Training set", "Validation set"], loc="upper left")
plt.savefig("model-fit-accuracy")
plt.close()

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Training set", "Validation set"], loc="upper left")
plt.savefig("model-fit-loss")
plt.close()
"""