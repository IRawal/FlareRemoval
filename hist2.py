import numpy as np
from PIL import Image

# CHATGPTs implementation-- resulted in hightened contrast but
# not in the way that we want

image = Image.open('images/20240413_104245_HoloLens.jpg')

if image.mode != 'L':
    image = image.convert('L')

# image.show()

image_array = np.array(image)

histogram, _ = np.histogram(image_array.flatten(), bins=256, range=(0, 256))

cdf = histogram.cumsum()

linear_cdf = (cdf / cdf[-1]) * 255

equalized_image_array = linear_cdf[image_array.astype(int)]

equalized_image = Image.fromarray(equalized_image_array.astype(np.uint8))

equalized_image.show()

