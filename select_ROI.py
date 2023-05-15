import cv2

# original image directory
data_path_img  = 'C:\\FYP\\DeepGlobe-Road-Extraction-Challenge\\dataset\\train\\mask\\1L_l_1_vessels.png'
original_path_img = 'C:\\FYP\\DeepGlobe-Road-Extraction-Challenge\\dataset\\train\\image\\1L_l_1.jpg'


# saving directory 
output_directory_mask = 'C:\\FYP\\FYP_from_scratch\\output\\mask\\'
output_directory_image = 'C:\\FYP\\FYP_from_scratch\\output\\image\\'

# Load the image
img = cv2.imread(data_path_img)
original_img  = cv2.imread(original_path_img)


def get_output_name(name,image_number):
    
    splited_name = name.split("\\")
    output_name = splited_name[-1].split(".")
    if image_number==0:
        return 'crop_' + output_name[0]
    else:
        return str(image_number) + '_crop_' + output_name[0]


# Specify the window size
window_size = (800, 600)

# Create a named window with the specified size
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', window_size)

# Display the image in a window
cv2.imshow('Image', img)


def split_images(image,output_path,data_path_img):
    # Split the image into 256x256 sub-images
    sub_image_size = 256
    height, width, _ = image.shape

    sub_images = []
    for i in range(0, height, sub_image_size):
        for j in range(0, width, sub_image_size):
            sub_image = image[i:i+sub_image_size, j:j+sub_image_size]
            sub_images.append(sub_image)

    # Save the sub-images as separate images
    for i, sub_image in enumerate(sub_images):
        cv2.imwrite(output_path + get_output_name(data_path_img,0)+ f'_sub_image_{i+1}.jpg', sub_image)

# Mouse callback function
def get_clicked_pixel(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        pixel_list = param
        pixel_list.append((x, y))
        print(f"Clicked pixel: ({x}, {y})")
        
        crop_img = img[y:y+1024, x:x+1024]
        crop_original_img = original_img[y:y+1024, x:x+1024]
        split_images(crop_img,output_directory_mask,data_path_img)  
        split_images(crop_original_img,output_directory_image,original_path_img)
        # Concatenate the two images horizontally
        concatenated_img = cv2.hconcat([crop_img, crop_original_img])
        image_number = 0
        # cv2.imwrite(get_output_name(data_path_img,image_number), crop_img)
        # cv2.imwrite(get_output_name(original_path_img,image_number), crop_original_img)
        
        # Display the concatenated image
        cv2.imshow('Croped images', concatenated_img)


# Create an empty list to store the clicked pixel coordinates
clicked_pixels = []

# Set the mouse callback function with the list as the parameter
cv2.setMouseCallback('Image', get_clicked_pixel, clicked_pixels)

# Wait for a key press and then close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()