# python2
from numpy import *
from scipy.signal import convolve2d
from skimage.transform import resize

def get_histo(mod, atan):
	histo = zeros(9)

	for i in range(3):
		for j in range(3):
			for k in range(9):
				if atan[i, j] <= pi/9*(k + 1):
					break
			histo[k] += mod[i, j]

	return histo

def get_cells(mod, atan):
	cells = zeros((16,16,9))

	for i in range(16):
		for j in range(16):
			cells[i,j] = get_histo(mod[i*3 : (i+1)*3, j*3 : (j+1)*3], \
				atan[i*3 : (i+1)*3, j*3 : (j+1)*3])

	return cells

def normalization(vector): #Maybe other methods
	eps = 0.0001

	vector = vector/sqrt(vector.sum()**2  + eps)
	if (vector >= 0.2).sum() != 0:
		vector = (vector <= 0.2) * vector + (vector > 0.2) * 0.2
		vector = vector/sqrt(vector.sum()**2  + eps)

	return vector

def get_res(cells):
	res = []

	for i in range(15):
		for j in range(15):
			vector = hstack((cells[i, j], cells[i, j+1], \
				cells[i+1, j], cells[i+1, j+1]))
			norm = normalization(vector)
			res = hstack((res, norm))

	return res

def extract_hog(img, roi):
	roi = roi.astype(int)

	# separeting channels
	
	img_r = img[:,:,0]
	img_g = img[:,:,1]
	img_b = img[:,:,2]

	# cropping window

	img_r = img_r[roi[0] : roi[2] + 1, roi[1] : roi[3] + 1]
	img_g = img_g[roi[0] : roi[2] + 1, roi[1] : roi[3] + 1]
	img_b = img_b[roi[0] : roi[2] + 1, roi[1] : roi[3] + 1]

	# resize image to 48x48

	img_r = resize(img_r, (48,48))
	img_g = resize(img_g, (48,48))
	img_b = resize(img_b, (48,48))

	# make filters

	flt_x = array(([[-1, 0, 1]]))
	flt_y = array(([-1], [0], [1]))
#	flt_x = array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]))	# Maybe other filters(5 variants)
#	flt_y = array(([1, 2, 1], [0, 0, 0], [-1, -2, -1]))	#

	# derivative with filter
	
	img_derX_r = convolve2d(img_r, flt_x, mode = 'same', boundary = 'symm')
	img_derX_g = convolve2d(img_g, flt_x, mode = 'same', boundary = 'symm')
	img_derX_b = convolve2d(img_b, flt_x, mode = 'same', boundary = 'symm')
	img_derY_r = convolve2d(img_r, flt_y, mode = 'same', boundary = 'symm')
	img_derY_g = convolve2d(img_g, flt_y, mode = 'same', boundary = 'symm')
	img_derY_b = convolve2d(img_b, flt_y, mode = 'same', boundary = 'symm')

	# module of gradient

	modOfGrad_r = sqrt(img_derX_r**2 + img_derY_r**2)
	modOfGrad_g = sqrt(img_derX_g**2 + img_derY_g**2)
	modOfGrad_b = sqrt(img_derX_b**2 + img_derY_b**2)

	# direction of gradient

	arctanOfGrad_r = arctan2(img_derY_r, img_derX_r)
	arctanOfGrad_g = arctan2(img_derY_g, img_derX_g)
	arctanOfGrad_b = arctan2(img_derY_b, img_derX_b)
	arctanOfGrad_r = (arctanOfGrad_r < 0) * (pi + arctanOfGrad_r) + arctanOfGrad_r * (arctanOfGrad_r > 0)
	arctanOfGrad_g = (arctanOfGrad_g < 0) * (pi + arctanOfGrad_g) + arctanOfGrad_g * (arctanOfGrad_g > 0)
	arctanOfGrad_b = (arctanOfGrad_b < 0) * (pi + arctanOfGrad_b) + arctanOfGrad_b * (arctanOfGrad_b > 0)

	# receive cells

	cells_r = get_cells(modOfGrad_r, arctanOfGrad_r)
	cells_g = get_cells(modOfGrad_g, arctanOfGrad_g)
	cells_b = get_cells(modOfGrad_b, arctanOfGrad_b)

	# make bloks, normalization and receive HOG

	result = hstack((get_res(cells_r), get_res(cells_g), get_res(cells_b)))

	return result
