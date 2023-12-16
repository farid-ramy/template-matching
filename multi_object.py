import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img_rgb = cv.imread("./game.jpg")
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

template = cv.imread("./coin.jpg", cv.IMREAD_GRAYSCALE)

w, h = template.shape[::-1]
res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):
    cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB))
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(res, cmap="gray")
plt.title("Template Matching Result")

plt.show()
