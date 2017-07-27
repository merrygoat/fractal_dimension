from PIL import Image
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

def main():
    image_directory = "C:/Users/Peter/Documents/GitHub/fractal_dimension/images/"
    file_prefix = "slice"
    file_suffix = ".png"
    num_images = 0
    
    num_files, image_size = setup_load_images(num_images, image_directory, file_prefix, file_suffix)
    image_data = np.zeros((num_files, image_size[0], image_size[1]))
    
    for i in range(num_files):
            tmp_image = Image.open(image_directory + file_prefix + str(i) + file_suffix)
            if tmp_image.mode == "RGB":
                tmp_image = tmp_image.convert(mode='L')
            image_data[i] = np.array(tmp_image.copy())
    


def setup_load_images(num_images, image_directory, file_prefix, file_suffix):
    if num_images == 0:
        file_list = glob(image_directory + file_prefix + "*" + file_suffix)
        num_files = len(file_list)
        if num_files == 0:
            print("No files found.")
            raise KeyboardInterrupt  # Used  to stop execution (instead of sys.exit which kills ipython kernel)
    else:
        num_files = num_images

    image_size = Image.open(image_directory + file_prefix + '0' + file_suffix).size
        
    return num_files, image_size
	
main()