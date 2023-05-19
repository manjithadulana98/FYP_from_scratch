import cv2
import numpy as np

# # Load the retinal image
image = cv2.imread('C:\\FYP\\FYP_from_scratch\\datasets\\DRIVE\\training\\images\\29_training.tif')


def remove_bright_area(image, threshold):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create a binary mask where pixels above the threshold are set to white (255)
    mask = np.where(gray > threshold, 255, 0).astype(np.uint8)
    
    # # Apply the mask to the original image
    # result = cv2.bitwise_and(image, image, mask=mask)
    
    subtracted_image = cv2.subtract(gray, mask)
    
    return subtracted_image


def open_close (image , kernel_size ):
    # kernel_size = 10  # Adjust the kernel size as needed
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # Step 3: Perform opening operation
    opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # Step 4: Perform closing operation
    closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel)
    
    return closed_image

# Load the original image
# original_image = cv2.imread('path/to/image.jpg')

# Set the brightness threshold (adjust this value according to your needs)
brightness_threshold = 190

# Remove bright areas from the image
processed_image = remove_bright_area(image, brightness_threshold)


processed_image = open_close(processed_image, 10)

# Step 2: Take the inverse of the image
processed_image = 255 - processed_image

_, processed_image = cv2.threshold(processed_image, 200, 255, cv2.THRESH_BINARY)


binary_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)

# Step 3: Perform the subtraction
subtracted_image = cv2.subtract(image, binary_image_rgb)


# Display the original and processed images
cv2.imshow('Original Image', image)
cv2.imshow('Processed Image', subtracted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # Step 5: Display the original, opened, and closed images
# cv2.imshow("Original Image", image)
# cv2.imshow("Opened Image", opened_image)
# cv2.imshow("Closed Image", closed_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
