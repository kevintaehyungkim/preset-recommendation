'''
image.py

Used to extract and manipulate image data, and generate image features for models

TO-DO items:
 - enable multiprocessing/pool to batch extraction/generation
'''
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from multiprocessing import Pool
import numpy as np
from skimage import io
from skimage.color import rgb2lab, deltaE_cie76
from PIL import Image


# Colors
RGB_COLOR_MAP = {
	"BLACK": [0, 0, 0],
	"WHITE": [255, 255, 255],
	"RED": [255, 0, 0],
	"ORANGE": [255, 165, 0],
	"YELLOW":[255, 255, 0],
	"GREEN": [0, 128, 0],
	"AQUA": [0, 255, 255],
	"BLUE": [0, 0, 255],
	"PURPLE": [128, 0, 128],
	"MAGENTA": [255, 0, 255]
}


'''
Returns a sequence of RGB data stored in a tuple for each pixel in the image 

Parameters:
  image: Image object to extract pixel data from
  color: RGB color to extract 
	0 - Red
	1 - Green
	2 - Blue
	None - RGB (default)
'''
def extract_pixel_data(image, color=None):
	if color is not None:
		pixel_data = image.getdata(color)
		return pixel_data
	return image.getdata()


'''
Returns a histogram of the image as a list of pixel counts, 
one for each pixel value in the source image. 
The histogram for an RGB image will contain 768 values.

Parameters:
  image: Image object to extract pixel data from
'''
def extract_histogram_data(image):
	return image.histogram()


'''
Returns the intensity score of how closely a pixel is related to the provided color
Only accounts for pixels that are within a certain color distance threshold
'''
def get_color_value(image, base_color):
	pixel_count, total = 0, 0
	dist_arr = []

	image_size = image.size
	num_pixels = image_size[0]*image_size[1]

	image_rgb_data = image.getdata()

	for pixel in image_rgb_data:
		dist_arr.append(get_cie2000_difference(pixel, base_color))

	min_dist = min(dist_arr)-(10**-9)
	max_dist = max(dist_arr)+(10**-9)
	threshold = (max_dist-min_dist)/15.0

	for color_dist in dist_arr:
		if color_dist-min_dist < threshold:
			pixel_count += 1
			total += 100.0/color_dist

	return total/pixel_count


def get_cie2000_difference(color1, color2):
	# Normalize RGB values
	color1_rgb = sRGBColor(color1[0], color1[1], color1[2], is_upscaled=True)
	color2_rgb = sRGBColor(color2[0], color2[1], color2[2], is_upscaled=True)

	# Convert from RGB to Lab Color Space
	color1_lab = convert_color(color1_rgb, LabColor)
	color2_lab = convert_color(color2_rgb, LabColor)

	#Find the difference
	diff = delta_e_cie2000(color1_lab, color2_lab)
	return diff


if __name__ == '__main__':
  	test_image = Image.open('test2.jpg')
  	print("image loaded")

  	test_image.thumbnail((150, 150))
  	print("image pixel data compressed")

  	for color in RGB_COLOR_MAP.keys():
  		print(color + " :" + str(get_color_value(test_image, RGB_COLOR_MAP[color])))

  	# print(get_cie2000_difference([255,0,0], [230,0,0]))
  	# print(reduce_rgb_data(test_image.getdata(), test_image.size[0], test_image.size[1]))

	# print(test.format)
	# print(test.mode)
	# print(test.size)
	test_image.show()







