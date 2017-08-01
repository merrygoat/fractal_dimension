from PIL import Image
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.patches as patches


def plot_smooth(x, y, style):
    """function to fit and plot a power law with a smooth curve"""
    def powerlaw(x, b, c):
        return b*x**c
    xnew = np.linspace(x.min(), x.max(), 300)
    popt, pcov = curve_fit(powerlaw, x, y, p0=(1,2))
    plt.plot(xnew, powerlaw(xnew, *popt), style, lw=0.3)
    return popt


def main2d(image_name, num_boxes, average_threshold, minimum_threshold):
    """main function for determining the fractal dimension of 2d datasets"""

    tmp_image = Image.open(image_name)
    image_size = tmp_image.size
    tmp_image = np.array(tmp_image.copy())
    if np.max(tmp_image) > 256:
        tmp_image = tmp_image/256
    image_data = tmp_image

    dimensions = []

    for box_size in range(2, num_boxes):
        dimensions.append(count_boxes_2d(image_data, box_size, image_size, average_threshold, minimum_threshold))

    plot_dimension(num_boxes, dimensions)


def main3d(image_directory, file_prefix, file_suffix, num_images, num_boxes, average_threshold, minimum_threshold):
    """main function for determining the fractal dimension of 3D datasets"""

    image_size = setup_load_images(num_images, image_directory, file_prefix, file_suffix)
    image_data = np.zeros((image_size[2], image_size[0], image_size[1]))

    for i in range(image_size[2]):
        tmp_image = Image.open(image_directory + file_prefix + '{:04d}'.format(i) + file_suffix)
        tmp_image = np.array(tmp_image.copy())
        if np.max(tmp_image) > 256:
            tmp_image = tmp_image / 256
        image_data[i] = tmp_image

    dimensions = []

    for box_size in range(2, num_boxes):
        dimensions.append(count_boxes_3d(image_data, box_size, image_size, average_threshold, minimum_threshold))

    plot_dimension(num_boxes, dimensions)


def plot_dimension(num_boxes, dimensions):
    """ Plot filled boxes against box magnification. The gradient is the fractal dimension"""
    plt.loglog(np.arange(2, num_boxes), dimensions, 'x')
    gradient = plot_smooth(np.arange(2, num_boxes), dimensions, '-k')
    plt.xlabel("magnification")
    plt.ylabel("filled boxes")
    plt.text(5, 10, "Fractal dimension = " + '{:1.2f}'.format(gradient[1]))
    plt.show()
    np.savetxt("fractal_dimension.txt", dimensions)


def count_boxes_2d(image_data, box_size, image_size, average_threshold, minimum_threshold, debug=True):
    """ Box counting for 2d images"""
    box_x_size = int(image_size[0] / box_size)
    box_y_size = int(image_size[1] / box_size)
    box_area = box_x_size * box_y_size * 0.1
    filled_boxes = 0
    if debug:
        plt.imshow(image_data)
        myax = plt.gca()

    for x_box_num in range(0, box_size):
        for y_box_num in range(0, box_size):
            x_min = x_box_num * box_x_size
            x_max = ((x_box_num + 1) * box_x_size)-1
            y_min = y_box_num * box_y_size
            y_max = (y_box_num + 1) * box_y_size
            if np.average(image_data[x_min:x_max, y_min:y_max]) > average_threshold:
                if (np.count_nonzero(image_data[x_min:x_max, y_min:y_max] > minimum_threshold)) > box_area:
                    filled_boxes += 1
                    if debug:
                        myax.add_patch(patches.Rectangle((y_min+2, x_min+2), box_x_size-4, box_y_size-4, fill=False, edgecolor="green", linewidth=2))
                else:
                    if debug:
                        myax.add_patch(patches.Rectangle((y_min+2, x_min+2), box_x_size-4, box_y_size-4, fill=False, edgecolor="red", linewidth=2))
    plt.show()

    return filled_boxes


def count_boxes_3d(image_data, box_size, image_size, average_threshold, minimum_threshold):
    """box counting for 3d images"""
    box_x_size = int(image_size[0]/box_size)
    box_y_size = int(image_size[1]/box_size)
    box_z_size = int(image_size[2]/box_size)
    box_area = box_x_size * box_y_size * box_x_size*0.05

    filled_boxes = 0

    for x_box_num in range(0, box_size):
        for y_box_num in range(0, box_size):
            for z_box_num in range(0, box_size):
                x_min = x_box_num * box_x_size
                x_max = ((x_box_num + 1) * box_x_size)-1
                y_min = y_box_num * box_y_size
                y_max = (y_box_num + 1) * box_y_size
                z_min = z_box_num * box_z_size
                z_max = (z_box_num + 1) * box_z_size
                if np.average(image_data[z_min:z_max, x_min:x_max, y_min:y_max]) > average_threshold:
                    if (np.count_nonzero(image_data[z_min:z_max, x_min:x_max, y_min:y_max] > minimum_threshold)) > box_area:
                        filled_boxes += 1
    return filled_boxes


def setup_load_images(num_images, image_directory, file_prefix, file_suffix):
    """ called for 3d datatsets, this globs theimage directory and assumes that all files
    fitting the search pattern are part of the dataset. It then loads all of these images
    into a numpy array."""
    if num_images == 0:
        file_list = glob(image_directory + file_prefix + "*" + file_suffix)
        num_files = len(file_list)
        if num_files == 0:
            print("No files found.")
            raise KeyboardInterrupt  # Used  to stop execution (instead of sys.exit which kills ipython kernel)
    else:
        num_files = num_images

    image_size = Image.open(image_directory + file_prefix + '0000' + file_suffix).size
    image_size = [image_size[0], image_size[1], num_files]

    return image_size