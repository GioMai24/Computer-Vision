import pandas as pd
from os import scandir
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('data/MEL/ISIC_0000040_downsampled.jpg')


img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(img[..., 0].flatten(), img[..., 1].flatten(), img[..., 2].flatten())
plt.show()