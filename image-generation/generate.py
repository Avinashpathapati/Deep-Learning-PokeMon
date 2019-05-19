# Generate module
# Author: Andreas Pentaliotis
# Module to implement image generation.

from keras.models import load_model

from utility import save, parse_input_arguments, generate_images


arguments = parse_input_arguments()
generator = load_model("../results/pikachu/epoch-1000/generator.h5")
images = generate_images(generator, int(arguments["number"]))
save(images, "./output")
