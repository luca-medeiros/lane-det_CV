import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read in and grayscale the image
image = mpimg.imread('exit-ramp.jpg')
print(image)
plt.imshow(image)
plt.show()