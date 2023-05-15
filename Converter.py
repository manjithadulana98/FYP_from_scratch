import os
import shutil
folder_path = "C:\\Users\\lanka\\Downloads\\SBVPI\\SBVPI"
train_path = "C:\\FYP\\DeepGlobe-Road-Extraction-Challenge\\dataset\\train\\mask"
image_path = "C:\\FYP\\DeepGlobe-Road-Extraction-Challenge\\dataset\\train\\image"


def convert(t):
    comma='.'
    
    words = t.split('.')
    first_word = words[0]

    return comma.join([first_word[:-8],'jpg'])

# Get a list of file names in the folder
file_names = os.listdir(folder_path)
separator = '\\'

for i in file_names:
    sub_file_names = separator.join([folder_path,i])
    sub_files = os.listdir(sub_file_names)
    for j in sub_files:
        if "vessels" in j :
            copy_folder = separator.join([sub_file_names,j])
            paste_folder =  separator.join([train_path,j])
            # Copy the file to the destination folder
            shutil.copy(copy_folder, paste_folder)
            
            image_data = convert(j)
            
            image_copy_folder = separator.join([sub_file_names,image_data])
            image_paste_folder =  separator.join([image_path,image_data])
            # Copy the file to the destination folder
            shutil.copy(image_copy_folder, image_paste_folder)
            
            # print(j)