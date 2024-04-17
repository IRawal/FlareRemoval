from PIL import Image as im
import numpy as np
from matplotlib import pyplot as plt
import pywt

# Alex's implementation -- helped to learn the idea behind the CDF linearization
# and general principles-- my python skills were simply holding me back

# img = np.asarray(im.open('images/20240413_104245_HoloLens.jpg').convert('L'))

# im.fromarray(img.save('image.jpg'))
# plt.imshow(img, interpolation='none')
# plt.show()

# img = im.open('images/20240413_104245_HoloLens.jpg').convert('L')
# img.save('grayscale.png')

img = im.open('images/20240413_104319_HoloLens.jpg')
gray_img = img.convert('L')
histogram = gray_img.histogram()

grayArr = np.zeros(256, dtype=float)
cdfx = np.zeros(256, dtype=float)
grayArr2 = np.zeros(256, dtype=float)
hist = np.zeros(256, dtype=float)

for i in range(256):
    grayArr[i] = histogram[i]
    grayArr[i] = grayArr[i] / 2147200
    cdfx[i] = grayArr[i] + cdfx[i-1]
    # grayArr2[i] = cdfx[i] * 2147200
    # hist[i] = int((cdfx[i] / cdfx[i-1]) * 255)

hist = (cdfx / cdfx[-1]) * 255

# Plot the histogram
plt.figure(figsize=(8, 6))
plt.hist(range(256), bins=256, weights=hist, color='black', alpha=0.7)
plt.title('Grayscale Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()

# print(gray_img)




