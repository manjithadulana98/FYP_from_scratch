import cv2
import numpy as np

# Step 1: Read the image
image = cv2.imread('image_vessels.png')

# Step 2: Preprocess the image (if necessary)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)

# Step 3: Find contours of the curve
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 4: Extract the largest contour (assumed to be the curve)
largest_contour = max(contours, key=cv2.contourArea)

# Step 5: Get the curve pixels
curve_pixels = np.squeeze(largest_contour)

# Step 6: Calculate the differences between consecutive y-values
differences = np.diff(curve_pixels[:, 1])

# Step 7: Calculate the derivatives
derivatives = differences / np.diff(curve_pixels[:, 0])

print(derivatives)

# Step 8: Find the indices of the stationary points where the derivative is approximately zero
stationary_point_indices = np.where(np.isclose(derivatives, 0.0))

print(stationary_point_indices)

# Step 9: Mark the stationary points on the image
marked_image = image.copy()
for pixels in curve_pixels:
    x, y = pixels
    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

# Step 10: Display the marked image
cv2.imshow("Marked Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
