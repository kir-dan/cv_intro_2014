# python2
from numpy import *
from scipy.signal import convolve2d
from skimage.transform import resize
from skimage.exposure import rescale_intensity
from skimage.filter import gaussian_filter, threshold_otsu, rank
from skimage.morphology import binary_closing, disk, label
from skimage.measure import regionprops
from skimage.io import imread, imsave
from glob import iglob	

def generate_template(digit_dir_path):
	template = zeros((42, 42))

	for filename in iglob(digit_dir_path + '/*.bmp'):
		img = imread(filename)
		img = resize(img, [44, 44])
		img = img[1:43,1:43]
		img = (img > (threshold_otsu(img, nbins=256)))
		template += img

	template = (template < (threshold_otsu(template, nbins=256)-5))
	return template

def max_area(props):
	if len(props) != 0:
		max_a = props[0].area
		for i in range(len(props)):
			if max_a < props[i].area:
				max_a = props[i].area
		return max_a
	else:
		return 0

def mse(img1, img2):
	img1.astype(int)
	img2.astype(int)
	return ((img1-img2)**2).sum()	

def bboxKey(prop):
	return prop.bbox[1]

# table selection
def table(img):

	img = rescale_intensity(img)
	img = ssr(img, 3)
	img = rank.median(img, disk(2))
	img = img[img.shape[0]*0.02:img.shape[0]*0.95, img.shape[1]*0.05:img.shape[1]*0.95]
	height = img.shape[0]
	width = img.shape[1]

	img = (img < threshold_otsu(img[height/5:height/5*4, width/5:width/5*4], nbins = 256))

	# Horizontal

	whiteList = []

	for i in range((height/2), height-4):
		img_lbl = label(img[i:i+4, 0:width], 4, background = 255)
		props = regionprops(img_lbl)
		whiteList.append(max_area(props))

	max_wh = max(whiteList)
	bool = 0

	for i in arange(len(whiteList))[::-1]:
		if (whiteList[i] > (0.95*max_wh)) and (bool == 0):
			h_top = i
			bool = 1

	whiteList = []

	for i in  arange(4, height/2)[::-1]:
		img_lbl = label(img[i-4:i, 0:width], 4, background = 255)
		props = regionprops(img_lbl)
		whiteList.append(max_area(props))

	max_wh = max(whiteList)
	bool = 0

	for i in arange(len(whiteList))[::-1]:
		if (whiteList[i] > (0.95*max_wh)) and (bool == 0):
			h_bot = i
			bool = 1

	img = img[(height/2) - h_bot:(height/2) + h_top,] * 255
	x_min = (height/2) - h_bot
	x_max = (height/2) + h_top

	# Vertical

	whiteList = []

	for i in range((width/2), width):
		whiteList.append(img[height/4:height/4*3, i].sum())

	max_wh = max(whiteList)
	bool = 0

	for i in arange(len(whiteList))[::-1]:
		if (whiteList[i] > (0.95*max_wh)) and (bool == 0):
			v_top = i
			bool = 1

	whiteList = []

	for i in arange(width/2)[::-1]:
		whiteList.append(img[height/4:height/4*3,i].sum())

	max_wh = max(whiteList)
	bool = 0

	for i in arange(len(whiteList))[::-1]:
		if (whiteList[i] > (0.95*max_wh)) and (bool == 0):
			v_bot = i
			bool = 1

	y_min = (width/2) - v_bot
	y_max = (width/2) + v_top

	if float(v_top + v_bot)/height < 1.5:
		y_min = 0
		y_max = width

	return (y_min, x_min, y_max, x_max)

# single-scale retinex 
def ssr(img, sig):
	img = img.astype(float)
	limg = gaussian_filter(img, sigma = sig)
	limg = limg + (limg == 0)
	img = img + (img == 0)
	img = (255*(log(img) - log(limg)) + 127)
	tmp = ones((img.shape[0], img.shape[1]), dtype = float)*255
	tmp = (img > 255) * tmp
	img = ((img >= 0)*(img <= 255)*img + tmp).astype(uint8)
	return img

# unsharp filter
def unsharp(img, a):
	img = img.astype(float)
	flt = array([[0.003, 0.013, 0.022, 0.013, 0.003], [0.013, 0.059, 0.097, 0.059, 0.013], [0.022, 0.097, 0.159, 0.097, 0.022], \
		[0.013, 0.059, 0.097, 0.059, 0.013], [0.003, 0.013, 0.022, 0.013, 0.003]])
	img = img + a*(img - convolve2d(img, flt, mode = 'same', boundary = 'symm'))
	tmp = ones((img.shape[0], img.shape[1]), dtype = float)*255
	tmp = (img > 255) * tmp
	img = ((img >= 0)*(img <= 255)*img + tmp).astype(uint8)
	return img

#from processed image give 3 digits 
# or (-1, -1, -1), if imposiible distinguish its
def giveNumbers(img, digit_templates, needClosing, delta):
	height = img.shape[0]
	width = img.shape[1]

	img = (img > (threshold_otsu(img) + delta))

	imsave('tmp.bmp', img * 255)

	if needClosing == 1:
		img = binary_closing(img, disk(1))

	img_lbl = label(img, 4, background = 255)

	props = regionprops(img_lbl)

	#delete small trash

	n_props = []
	for i in range(len(props)):
		if (float(props[i].bbox[3]-props[i].bbox[1])/(props[i].bbox[2]-props[i].bbox[0]) > 0.2) and \
			(float(props[i].bbox[3]-props[i].bbox[1])/(props[i].bbox[2]-props[i].bbox[0]) < 0.8) and \
			(props[i].area > ((height*0.1)*(width*0.1))) :
			n_props.append(props[i])

	#delete non-symbols trash

	props = []
	for i in range(len(n_props)):
		if compNum(n_props[i].image, digit_templates)[0] < 0.3:
			props.append(n_props[i])

	props.sort(key = bboxKey)

	#delete letters

	n_props = []
	max_h = 0
	for i in range(len(props)):
		if props[i].bbox[2] - props[i].bbox[0] > max_h:
			max_h = props[i].bbox[2] - props[i].bbox[0]

	n_props = []
	for i in range(len(props)):
		if props[i].bbox[2] - props[i].bbox[0] > 0.85*max_h:
			n_props.append(props[i])

	# maube not all digits in props, so return (-1, -1, -1)

	if len(n_props) < 3:
		return (-1, -1, -1) 

	lenNum = n_props[0].bbox[3] - n_props[0].bbox[1]

	props = [n_props[0]]
	bool = 0
	for i in range(1, len(n_props)):
		if n_props[i].bbox[1] - n_props[i-1].bbox[3] < lenNum \
			and bool == 0:
			props.append(n_props[i])
		else:
			bool = 1

	if len(props) >= 3:
		return (compNum(props[0].image, digit_templates)[1], \
			compNum(props[1].image, digit_templates)[1], \
			compNum(props[2].image, digit_templates)[1])
	else:
		return (-1, -1, -1)

# comparison of img and templates
# return (how mach they are like, what number)
def compNum(img, digit_templates):
	num = 0
	x = img.shape[0]
	y = img.shape[1]
	min = x * y
	for i in range(len(digit_templates)):
		tmp = resize(digit_templates[i], [x, y]).astype(uint8)
		if (min > mse(img, tmp)):
			min = mse(img, tmp)
			num = i
			compMetric = mse(img, tmp) / (x * y)
	return (compMetric, num)

def recognize(img, digit_templates):
	
	# table selection

	table_coord = table(img)
	img = img[img.shape[0]*0.02:img.shape[0]*0.95, \
	img.shape[1]*0.05:img.shape[1]*0.95]
	img = img[table_coord[1]:table_coord[3], \
	table_coord[0]:table_coord[2]]

	# end of table selection

	st_img = img.copy()

	# image processing

	img = rescale_intensity(img)
	img = rank.median(img, disk(2))

	img = ssr(img, 9)
	img = unsharp(img, 3)

	# end of image processing

	result = giveNumbers(img, digit_templates, 0, 0)


	# numbers very near: without median(else connect) 
	# and low sigma in gaussian in ssr(else connect too)

	if result == (-1, -1, -1):
		img = st_img.copy()
		img = rescale_intensity(img)
		img = ssr(img, 3)
		img = unsharp(img, 3)
		imsave('tmp.bmp', img)
		result = giveNumbers(img, digit_templates, 0, 0)
	

	# very bright picture: need closing in giveNumbers

	if result == (-1, -1, -1):
		img = st_img.copy()
		img = rescale_intensity(img)
		img = rank.median(img, disk(2))
		img = ssr(img, 9)
		img = unsharp(img, 3)
		result = giveNumbers(img, digit_templates, 1, 0)
	

	# and play with threshold of binarization

	delta = 0
	bool = -1
	while result == (-1, -1, -1) and abs(delta) <= (240):
		delta *= -1
		if bool == -1:
			delta -= 5
		bool *= -1
		img = st_img.copy()
		img = rescale_intensity(img)
		img = rank.median(img, disk(2))
		img = ssr(img, 9)
		img = unsharp(img, 3)
		result = giveNumbers(img, digit_templates, 0, delta)

	if result == (-1, -1, -1):
		result = (0, 0, 0)


	return result
