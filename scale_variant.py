import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("./messi.jpg")
template = cv2.imread("./messi_face.jpg")
# Convert images to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

match_results = []

# Define the scale factors for resizing the template
scale_factors = np.linspace(0.2, 1.0, 10)[::-1]
print(scale_factors)

for scale in scale_factors:
    # Resize the template
    resized_template = cv2.resize(template_gray, None, fx=scale, fy=scale)

    # Perform template matching
    res = cv2.matchTemplate(img_gray, resized_template, cv2.TM_CCOEFF_NORMED)

    # Find the maximum correlation value and its location
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # Calculate the top-left and bottom-right coordinates of the matched region
    w, h = resized_template.shape[::-1]
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Store the matched result and scale information
    match_results.append((max_val, top_left, bottom_right, scale))

# Sort the match results based on the correlation value in descending order
match_results.sort(reverse=True)

# Get the best match (highest correlation value)
best_match = match_results[0]
max_val, top_left, bottom_right, scale = best_match

# Rescale the top-left and bottom-right coordinates to the original image scale
top_left = (int(top_left[0] / scale), int(top_left[1] / scale))
bottom_right = (int(bottom_right[0] / scale), int(bottom_right[1] / scale))

# Draw a rectangle around the matched region on the original image
cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Scale Variant Template Matching")
plt.show()
