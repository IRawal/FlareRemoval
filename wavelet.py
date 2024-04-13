from PIL import Image as im
import numpy as np
import pywt


img = np.asarray(im.open('images/20240413_104245_HoloLens.jpg').convert('L'))
trans = pywt.dwt2(img, 'db1')
print(trans)