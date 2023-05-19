import cv2
import numpy as np

# Global variables
selected_pixel = None

def mouse_callback(event, x, y, flags, param):
    global selected_pixel

    if event == cv2.EVENT_LBUTTONDOWN:
        selected_pixel = (x, y)
        remove_circle()

def remove_circle():
    global selected_pixel

    if selected_pixel is not None:
        circle_radius = 50
        circle_thickness = -1  # Negative thickness fills the circle
        circle_color = (0, 0, 0)  # Black color for the circle

        # Draw the circle on a separate mask image
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, selected_pixel, circle_radius, 255, circle_thickness)

        # Apply the mask to the image to remove the circle
        result = cv2.bitwise_and(image, image, mask=mask)
        subtracted_image = cv2.subtract(image, result)

        # Display the resulting image
        cv2.imshow("Image with Circle Removed", subtracted_image)

# Read the image
image = cv2.imread('C:\\FYP\\FYP_from_scratch\\datasets\\DRIVE\\training\\images\\29_training.tif')

# Create a window and set the mouse callback function
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_callback)

# Display the image
cv2.imshow("Image", image)

# Wait for a key press to exit
cv2.waitKey(0)
cv2.destroyAllWindows()
