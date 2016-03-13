from numpy import roll, dstack, array
from skimage.transform import rescale

def mse(i1, i2):
	return ((i1 - i2)**2).sum()


def kor(i1, i2):
	return ((i1*i2).sum())

def half_image(i1, k):
	if k != 0:
		for i in range(k):
			i1 = i1[::2,::2]
	return i1

def align(bgr_image):
	print 'start'

	width = bgr_image.shape[1]
	height = bgr_image.shape[0]/3

	vdelta = height * 0.05
	hdelta = width * 0.05

	image1 = bgr_image[vdelta:height-vdelta, hdelta:width-hdelta].astype(int)
	image2 = bgr_image[height+vdelta:2*height-vdelta, hdelta:width-hdelta].astype(int)
	image3 = bgr_image[2*height+vdelta:3*height-vdelta, hdelta:width-hdelta].astype(int)

	tmpwidth = width
	tmpheight = height
	steps = 0;
	while tmpwidth > 500 and tmpheight > 500:
		steps += 1
		tmpwidth /= 2;
		tmpheight /= 2;

	steps += 1

	if steps > 2:
		interval = range(2, steps)[::-1]
	else:
		interval = range(steps-1, steps)[::-1]

	#----------------------------------------------------------------------

	for k in interval:
		cur_image1 = half_image(image1, k)
		cur_image2 = half_image(image2, k)
		cur_image3 = half_image(image3, k)
		border = 8/(2**(steps-k-1))
		tmp_cur_image2 = cur_image2[border:cur_image2.shape[0]-border, border:cur_image2.shape[1]-border]

		max = -1
		posi = 0
		posj = 0
		
		for i in range((-1)*border/2, border+1, 1):
			for j in range((-1)*border/2, border+1, 1):
				tmp_cur_image3 = roll(roll(cur_image3, i, axis = 0), j, axis = 1)
				tmp_cur_image3 = tmp_cur_image3[border:tmp_cur_image3.shape[0]-border, border:tmp_cur_image3.shape[1]-border]
				if kor(tmp_cur_image2, tmp_cur_image3) > max:
					max = kor(tmp_cur_image2, tmp_cur_image3)
					posi = i
					posj = j

		image3 = roll(roll(image3, (posi*(2**k)), axis = 0), (posj*(2**k)), axis = 1)

		max = -1
		posi = (-1) * border
		posj = (-1) * border
		
		for i in range((-1)*border, (border+1)/2, 1):
			for j in range((-1)*border, (border+1)/2, 1):
				tmp_cur_image1 = roll(roll(cur_image1, i, axis = 0), j, axis = 1)
				tmp_cur_image1 = tmp_cur_image1[border:tmp_cur_image1.shape[0]-border, border:tmp_cur_image1.shape[1]-border]
				if kor(tmp_cur_image2, tmp_cur_image1) > max:
					max = kor(tmp_cur_image2, tmp_cur_image1)
					posi = i
					posj = j

		image1 = roll(roll(image1, (posi*(2**k)), axis = 0), (posj*(2**k)), axis = 1)
	#----------------------------------------------------------------------

	image1 = image1[vdelta:height-3*vdelta, hdelta:width-3*hdelta]
	image2 = image2[vdelta:height-3*vdelta, hdelta:width-3*hdelta]
	image3 = image3[vdelta:height-3*vdelta, hdelta:width-3*hdelta]

	res_image = dstack((image3, image2, image1))
	print 'end'

	return res_image
