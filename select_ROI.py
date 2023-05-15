import cv2


data_path_img  = 'C:\\FYP\\DeepGlobe-Road-Extraction-Challenge\\dataset\\train\\mask\\1L_l_1_vessels.png'
original_path_img = 'C:\\FYP\\DeepGlobe-Road-Extraction-Challenge\\dataset\\train\\image\\1L_l_1.jpg'
# Load the image
img = cv2.imread(data_path_img)
original_img  = cv2.imread(original_path_img)


def get_output_name(name,image_number):
    
    splited_name = name.split("\\")
    if image_number==0:
        return 'crop_' + splited_name[-1]
    else:
        return str(image_number) + '_crop_' + splited_name[-1]


# Specify the window size
window_size = (800, 600)

# Create a named window with the specified size
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', window_size)

# Display the image in a window
cv2.imshow('Image', img)

# Mouse callback function
def get_clicked_pixel(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        pixel_list = param
        pixel_list.append((x, y))
        print(f"Clicked pixel: ({x}, {y})")
        
        crop_img = img[y:y+512, x:x+512]
        crop_original_img = original_img[y:y+512, x:x+512]
        
        # Concatenate the two images horizontally
        concatenated_img = cv2.hconcat([crop_img, crop_original_img])
        image_number = 0
        cv2.imwrite(get_output_name(data_path_img,image_number), crop_img)
        cv2.imwrite(get_output_name(original_path_img,image_number), crop_img)
        
        # Display the concatenated image
        cv2.imshow('Croped images', concatenated_img)


# Create an empty list to store the clicked pixel coordinates
clicked_pixels = []

# Set the mouse callback function with the list as the parameter
cv2.setMouseCallback('Image', get_clicked_pixel, clicked_pixels)

# Wait for a key press and then close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()


