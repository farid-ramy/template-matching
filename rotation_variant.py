import cv2
import numpy as np
from matplotlib import pyplot as plt


def rotate_image(image, angle):
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)


img = cv2.imread("./messi_flip.jpg")
template = cv2.imread("./messi_face.jpg")

# Convert images to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

match_results = []

# Define the rotation angles in degrees
rotation_angles = np.arange(0, 360, 10)
print(rotation_angles)

for angle in rotation_angles:
    # Rotate the template
    rotated_template = rotate_image(template_gray, angle)

    # Perform template matching
    res = cv2.matchTemplate(img_gray, rotated_template, cv2.TM_CCOEFF_NORMED)

    # Find the maximum correlation value and its location
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # Calculate the top-left and bottom-right coordinates of the matched region
    w, h = rotated_template.shape[::-1]
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Store the matched result and rotation information
    match_results.append((max_val, top_left, bottom_right, angle))

# Sort the match results based on the correlation value in descending order
match_results.sort(reverse=True)

# Get the best match (highest correlation value)
best_match = match_results[0]
max_val, top_left, bottom_right, angle = best_match

# Rotate the template to the best matching angle
best_rotated_template = rotate_image(template_gray, angle)

# Draw a rectangle around the matched region on the original image
cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)

plt.subplot(121), plt.imshow(cv2.cvtColor(best_rotated_template, cv2.COLOR_GRAY2RGB))
plt.title("Best Rotated Template")
plt.subplot(122), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Rotation Variant Template Matching")
plt.show()
