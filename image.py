'''
image.py

Used to extract and manipulate image data, and generate image features for models
'''

import colorsys
import io
import math
import time
import psutil
import ray
import scipy.signal
import numpy as np
import cython
import matplotlib.pyplot as plt

from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from PIL import Image

# NUMBER CPUS
NUM_CPUS = psutil.cpu_count(logical=False)

# Colors
RGB_COLOR_MAP = {
	# "WHITE": [255,255,255],
	# "LMAO": [250,255,255],
	# "AZURE": [240,255,255],
	# "LIGHT_CYAN": [224,255,255],



	"RED": [255,0,0],
	"DARK_RED": [139,0,0],
	"ORANGE": [255,165,0],
	"ORANGE_RED": [255,69,0],
	"DARK_ORANGE": [255,140,0],
	"DARK_GOLDEN_ROD": [184,134,11],
	"GOLDEN_ROD": [218,165,32],
	"GOLD": [255,215,0],
	"WHITE": [255,255,255],
	"LMAO": [250,255,255],
	"AZURE": [240,255,255],
	"LIGHT_CYAN": [224,255,255],
	"YELLOW":[255,255,0],
	"YELLOW_GREEN": [154,205,50],
	"GREEN": [0,128,0],
	"DARK_GREEN": [0,100,0],
	"AQUA": [0,255,255],
	"TEAL": [0,128,128],
	"BLUE": [0,0,255],
	"MEDIUM_BLUE": [0,0,205],
	"NAVY": [0,0,128],
	"PURPLE": [128,0,128],
	"MAGENTA": [255,0,255],
	"DARK_SLATE_GRAY": [47,79,79],
	"BLACK": [0,0,0]
}

RGB_COLOR_KEYS = [
	"WHITE",
	"LMAO",
	"AZURE",
	"LIGHT_CYAN",
	"RED",
	"DARK_RED",
	"ORANGE_RED",
	"DARK_GOLDEN_ROD",
	"ORANGE",
	"DARK_ORANGE",
	"GOLDEN_ROD",
	"GOLD",
	"YELLOW",
	"YELLOW_GREEN",
	"GREEN",
	"DARK_GREEN",
	"AQUA",
	"TEAL",
	"BLUE",
	"MEDIUM_BLUE",
	"NAVY",
	"PURPLE",
	"MAGENTA",
	"DARK_SLATE_GRAY",
	"BLACK"
	]

# ray.init(num_cpus=NUM_CPUS)


'''
Returns a sequence of RGB data stored in a tuple for each pixel in the image 
----------
Parameters
  image: Image object to extract pixel data from
  color: RGB color to extract 
    0 - Red
    1 - Green
    2 - Blue
    None - RGB (default)
----------
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
----------
Parameters
  image: Image object
----------
'''
def extract_image_histogram(image):
	return np.asarray(image.histogram())


'''
Returns a map of image color to color score
----------
Parameters
  image: Image object
----------
'''
def map_image_color_scores(image):
	image_data_arr = image.getdata()

	p = ThreadPool(2*NUM_CPUS)
	zzz = [[image_data_arr,color_key] for color_key in RGB_COLOR_MAP.keys()]
	color_dist_arr=p.map(get_image_color_score, zzz)

	# print(color_dist_arr)

	xd = {}
	for zzz in color_dist_arr:
		xd[zzz.keys()[0]] = zzz.values()[0]

	print(xd)

	xp = [xd[c] for c in RGB_COLOR_KEYS]
	return xp

  	# return color_dist_arr



def map_image_color_score(image_data_arr, index):
	# print(image)
	# image_data_arr = image.getdata()

	# p = ThreadPool(2*NUM_CPUS)
	# zzz = [[image_data_arr,RGB_COLOR_MAP.keys()[index]]]
	# color_dist_arr=p.map(get_image_color_score, zzz)
	color_score = get_image_color_score([image_data_arr, RGB_COLOR_MAP.keys()[index]])
  	return color_score


'''
Returns the intensity score of how closely a pixel is related to the provided color
Algorithm only accounts for pixels within a certain CIE-2000 color distance apart from the threshold
----------
Parameters
  image: Image object
  base_color: tuple of the RGB values for color given
----------
'''
def get_image_color_score(image_color_data):
	pixel_count, total = 0, 0
	min_max_ratio = 5.0 # controls acceptable range to classify as given color
	dist_value = 1.5 # controls inverse distance multiplier (closer to color -> more intense) 

	image_rgb_data = image_color_data[0]
	color_key = image_color_data[1]
	base_color = RGB_COLOR_MAP[color_key]

	p = Pool(2*NUM_CPUS)
	combined_pixels_arr = [[i,base_color] for i in image_rgb_data]
	color_dist_arr=p.map_async(get_cie2000_difference, combined_pixels_arr).get()

	min_dist = min(color_dist_arr)
	max_dist = max(color_dist_arr)
	threshold = (max_dist-min_dist)/min_max_ratio

	for color_dist in color_dist_arr:
		if color_dist-min_dist < threshold:
			pixel_count += 1
			total += 20.0/(color_dist+dist_value)

	color_score = total/pixel_count
	return {color_key: color_score}



'''
Returns the color difference between two RGB values
----------
Parameters
  color1: RGB tuple of first color
  color2: RGB tuple of second color
----------
'''
def get_cie2000_difference(color_arr):
	color1=color_arr[0]
	color2=color_arr[1]
	# Normalize RGB values
	color1_rgb = sRGBColor(color1[0], color1[1], color1[2], is_upscaled=True)
	color2_rgb = sRGBColor(color2[0], color2[1], color2[2], is_upscaled=True)

	# Convert from RGB to Lab Color Space
	color1_lab = convert_color(color1_rgb, LabColor)
	color2_lab = convert_color(color2_rgb, LabColor)

	#Find the difference
	diff = delta_e_cie2000(color1_lab, color2_lab)
	return diff

'''
Returns a modified image rgb data sequence provided color intensity scores

TO-DO: modify HIV intensity as well
----------
Parameters 
  rgb_data: [(255,100,100), (100,255,100), (0,0,255) ...]
  color_score_map: {"BLACK": 4.534, "WHITE": 3.1415 ...}
----------
'''
def modify_rgb_from_color_score(rgb_data, color_score_map):
	new_rgb_data = []
	for pixel in rgb_data:
		new_rgb_sum = [0, 0, 0]
		# print("hsv")
		# print(rgb_to_hsv(pixel))
		# print("hls")
		# print(rgb_to_hls(pixel))
		for color_key in color_score_map.keys():
			color = RGB_COLOR_MAP[color_key]
			color_dist = get_cie2000_difference(color, pixel)
			color_score = color_score_map[color_key]
			min_color_score = int(min(color_score_map.values()))

			r_diff = color[0]-pixel[0]
			g_diff = color[1]-pixel[1]
			b_diff = color[2]-pixel[2]

			if color_dist < 10:
				new_rgb_sum[0] += calculate_pixel_delta(r_diff, color_dist, color_score, min_color_score, 6.0)
				new_rgb_sum[1] += calculate_pixel_delta(g_diff, color_dist, color_score, min_color_score, 6.0)
				new_rgb_sum[2] += calculate_pixel_delta(b_diff, color_dist, color_score, min_color_score, 6.0)
			elif color_dist < 25:
				new_rgb_sum[0] += calculate_pixel_delta(r_diff, color_dist, color_score, min_color_score, 3.5)
				new_rgb_sum[1] += calculate_pixel_delta(g_diff, color_dist, color_score, min_color_score, 3.5)
				new_rgb_sum[2] += calculate_pixel_delta(b_diff, color_dist, color_score, min_color_score, 3.5)
			else:
				new_rgb_sum[0] += calculate_pixel_delta(r_diff, color_dist, color_score, min_color_score, 3.1)
				new_rgb_sum[1] += calculate_pixel_delta(g_diff, color_dist, color_score, min_color_score, 3.1)
				new_rgb_sum[2] += calculate_pixel_delta(b_diff, color_dist, color_score, min_color_score, 3.1)

		r_new = pixel[0] + new_rgb_sum[0]
		g_new = pixel[1] + new_rgb_sum[1]
		b_new = pixel[2] + new_rgb_sum[2]

		r_adjusted = max(max(0,r_new), min(r_new, 255))
		g_adjusted = max(max(0,g_new), min(g_new, 255))
		b_adjusted = max(max(0,b_new), min(b_new, 255))

		new_rgb = [r_adjusted, g_adjusted, b_adjusted]
		new_rgb_data.append(new_rgb)

	return new_rgb_data


def calculate_pixel_delta(pixel_color_diff, color_dist, color_score, base, exp):
	# print(int((pixel_color_diff/(1.0+(0.5*color_dist)))*(math.log(color_score,base)**exp)))
	return int((pixel_color_diff/(10.0*color_dist))*(math.log(color_score,base/1.5)**exp))


# return index of closest pixel 
def get_closest_base_color(rgb_pixel):
	return 0



def color_score_mean():
	return 0

'''
Generates a new Image object given RGB values
----------
Parameters 
  size: (width, height)
  rgb_data: [(255,100,100), (100,255,100), (0,0,255)... ]
----------
'''
def generate_image_rgb(size, rgb_data):
	rgb_data_flattened = [color_value for rgb_value in rgb_data for color_value in rgb_value] 
	new_image = Image.frombuffer('RGB', size, np.uint8(rgb_data_flattened))
	return new_image


def rgb_to_hsv(pixel_rgb):
	return colorsys.rgb_to_hsv(pixel_rgb[0], pixel_rgb[1], pixel_rgb[2])


def rgb_to_hls(pixel_rgb):
	return colorsys.rgb_to_hls(pixel_rgb[0], pixel_rgb[1], pixel_rgb[2])


def get_histogram_index(rgb_pixel):
	return 0


def round_number(x, base=25):
    return int(base * round(float(x)/base))


def serialize_rgb(r,g,b):
	return str(r) + "." + str(g) + "." + str(b)

# need to optimize order
def get_color_values(rgb_bin):
	bin_size = np.arange(0,11)*25
	color_vals = []
	for r in bin_size:
		for g in bin_size:
			for b in bin_size:
				rgb_key = serialize_rgb(r,g,b)
				if rgb_key in rgb_bin.keys():
					color_vals.append(rgb_bin[rgb_key])
				else: 
					color_vals.append(0)
	return color_vals

def normalize_data(data_arr):
	return data_arr/np.linalg.norm(data_arr)


def run():
	test_image = Image.open('test3.jpg')
  	print("image loaded")

  	test_image.thumbnail((80, 80))
  	print("image pixel data compressed")
	return np.asarray(map_image_color_scores(test_image))


if __name__ == '__main__':
	# print("main")

	# u know how often these colors are used
	# given a pixel, how 

  	image_1 = Image.open('training_1.jpg')
  	image_2 = Image.open('training_2.jpg')
  	image_3 = Image.open('training_3.jpg')

  	training_images = [image_1, image_2, image_3]


  	red_histogram = np.zeros(256)
  	green_histogram = np.zeros(256)
  	blue_histogram = np.zeros(256)

  	print("training images loaded")

  	# for image in training_images:
  	# 	r,g,b = image.split()
  	# 	red_histogram += r.histogram()
  	# 	green_histogram += g.histogram()
  	# 	blue_histogram += b.histogram()

  	# red_histogram_normalized = red_histogram/np.linalg.norm(red_histogram)
  	# green_histogram_normalized = green_histogram/np.linalg.norm(green_histogram)
  	# blue_histogram_normalized = blue_histogram/np.linalg.norm(blue_histogram)


  	# plt.plot(red_histogram_normalized, 'r-', lw=1)
  	# plt.plot(green_histogram_normalized, 'g-', lw=1)
  	# plt.plot(blue_histogram_normalized, 'b-', lw=1)
  	# plt.show()

  	rgb_bucket = {}

  	for image in training_images:
  		image.thumbnail((400, 400))
  		image_data = image.getdata()
  		for rgb_pixel in image_data:
  			new_rgb_str = str(round_number(rgb_pixel[0])) + "." + str(round_number(rgb_pixel[1])) + "." + str(round_number(rgb_pixel[2]))
  			if new_rgb_str in rgb_bucket.keys():
  				rgb_bucket[new_rgb_str] += 1
  			else:
  				rgb_bucket[new_rgb_str] = 1

  	# print(rgb_bucket)

  	color_values = normalize_data(get_color_values(rgb_bucket))

  	plt.plot(color_values)
  	plt.show()






  	# test_image.thumbnail((200, 200))

  	# print("image pixel data compressed")

  	# red_histogram = extract_image_histogram(image_1)
  	# green_histogram_2 = extract_image_histogram(image_2)
  	# histogram_3 = extract_image_histogram(image_3)

  	# collective_histogram = histogram_1 + histogram_2 + histogram_3
  	# histogram_normalized = collective_histogram/np.linalg.norm(collective_histogram)


  	# # test_image.show()
  	# a = time.time()

  	# color_score_map = map_image_color_score(test_image)

  	# b = time.time()
  	# print(b-a)
  # 	color_score_map = {
  # 		"BLUE": 5.69960501734, 
  # 		"PURPLE": 5.50955807294,
		# "BLACK": 6.2956522456,
		# "YELLOW": 3.61478414384,
		# "ORANGE": 10.7373946324,
		# "GREEN": 4.24207603381,
		# "WHITE": 20.4800148018,
		# "AQUA": 5.08449534265,
		# "MAGENTA": 5.91711920443,
		# "RED": 6.90746590358}
  	# print(color_score_map)


  	# test_image2 = Image.open('test.jpg')
  	# print("image loaded")

  	# test_image2.thumbnail((80, 80))
  	# print("image pixel data compressed")

  	# new_rgb_data = modify_rgb_from_color_score(test_image2.getdata(), color_score_map)
  	# print("new rgb data obtained")

  	# new_img = generate_image_rgb(test_image2.size, new_rgb_data)
  	# new_img.show()










 #  	zzz = list(test_image.getdata())
 #  	new_z = [item for t in zzz for item in t] 

 #  	a =np.asarray(new_z)
 #  	b = np.asarray([15]*34567095)
 #  	new_zz = a + b

	# print(new_zz[0:10])	

 #  	image_recreated = Image.frombuffer(test_image.mode, test_image.size, np.uint8(new_zz))
 #  	image_recreated.show()




	# print(get_cie2000_difference([255,0,0], [230,0,0]))
	# print(reduce_rgb_data(test_image.getdata(), test_image.size[0], test_image.size[1]))

	# print(test.format)
	# print(test.mode)
	# print(test.size)
	# test_image.show()