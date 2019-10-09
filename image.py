'''
image.py

Used to extract and manipulate image data, and generate image features for models

TO-DO items:
 - enable multiprocessing/pool to batch extraction/generation
'''

from PIL import Image


# Colors
BLACK = [0, 0, 0]
WHITE = [255, 255, 255]

RED = [255, 0, 0]
ORANGE = [255, 165, 0]
YELLOW = [255, 255, 0]
GREEN = [0, 128, 0]
AQUA = [0, 255, 255]
BLUE = [0, 0, 255]
PURPLE = [128, 0, 128]
MAGENTA = [255, 0, 255]



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

def get_red_value(histogram_arr):
	return sum(histogram_arr[0:256])

def get_blue_value(histogram_arr):
	return sum(histogram_arr[256:512])

def get_green_value(histogram_arr):
	return sum(histogram_arr[512:768])


# def get_black_value(rgb_arr):
# 	for 





if __name__ == '__main__':
  	test = Image.open('test.jpg')
  	print("image loaded")
	# print(test.format)
	# print(test.mode)
	# print(test.size)
	# test.show()
	# a = list(test.getdata())
	# print(extract_pixel_data(test)[0])
	# print(extract_pixel_data(test, 0)[0])
	# print(extract_pixel_data(test, 1)[0])
	# print(extract_pixel_data(test, 2)[0])


	# print(list(test2_data))

	test_histogram = extract_histogram_data(test)
	print(get_red_value(test_histogram))
	print(get_blue_value(test_histogram))
	print(get_green_value(test_histogram))







