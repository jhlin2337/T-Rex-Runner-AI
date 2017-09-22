import numpy as np
import constants
from PIL import Image

# Arguments:
# rgba_image - RGBA PIL Image that we want to pass into the neural network
# Returns:
# greyscale_vector - 1D array of a resized and unrolled greyscale image representation
#  					 of the rgba_image input.
def rgbaImg2nnInput(rgba_image):
	thumbnail_size = constants.THUMBNAIL_WIDTH, constants.THUMBNAIL_HEIGHT

	greyscale_img = rgba_image.convert('L')
	greyscale_img.thumbnail(thumbnail_size, Image.ANTIALIAS)
	greyscale_array = np.asarray(greyscale_img)
	# Image.fromarray(greyscale_array, 'L').show()
	greyscale_vector = greyscale_array.reshape(greyscale_array.shape[0]*greyscale_array.shape[1], 1)

	return greyscale_vector