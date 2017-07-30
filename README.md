# Fractal Dimension

Determination of Hausdorf dimension using box counting.
2D images or 3D images composed of stacks of 2D images can be procesed. Images are converted to 8 bit greyscale if they are not already.
The Ipython notebook "example_usage.ipynb" contains examples of usage.

Parameters:
* image_directory -	Directory path of images to analyse, end with slash
* file_prefix     - Prefix for file. Program assumes files are numbered sequentially from 0 with 4 digits i.e. image_0000.png, image_0001.png...
* file_suffix     - Image file type, ".png" or ".tif" are know to work. Other formats untested. 
* num_images  -	Number of timesteps to load from disk. Set to 0 for all available images.
* num_boxes - the number of different magnification levels used for box counting. More will tend to provide a more accurate answer but will be slower.
* threshold - the average brightness a box has to be before it is considered filled. A suitable value will depend on the level of noise in the image. Lower values will give more accurate answers in general though if the value is set too low this will increase the number of spurious box detections which will tend to increase the measured fractal dimension.

To confirm correct operation, the file "example_triangle.jpg" should give a fractal dimension of approximately 1.58.