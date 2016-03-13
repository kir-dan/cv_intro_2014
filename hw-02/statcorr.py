from numpy import *

def convTo(img):
	l = matrix(([0.3811, 0.5783, 0.0402], \
		[0.1967, 0.7244, 0.0782], \
		[0.0241, 0.1288, 0.8444]))

	oimg = img.copy()
	img[:,:,0] = l[0,0]*oimg[:,:,0] + l[0,1]*oimg[:,:,1] + l[0,2]*oimg[:,:,2]
	img[:,:,1] = l[1,0]*oimg[:,:,0] + l[1,1]*oimg[:,:,1] + l[1,2]*oimg[:,:,2]
	img[:,:,2] = l[2,0]*oimg[:,:,0] + l[2,1]*oimg[:,:,1] + l[2,2]*oimg[:,:,2]

	img[:,:,0] = log(img[:,:,0]+1)
	img[:,:,1] = log(img[:,:,1]+1)
	img[:,:,2] = log(img[:,:,2]+1)

	a = array(([1/sqrt(3), 0, 0], [0, 1/sqrt(6), 0], [0, 0, 1/sqrt(2)]))
	b = array(([1, 1, 1], [1, 1, -2], [1, -1, 0])).astype(float)
	c = dot(a, b)

	oimg = img.copy()
	img[:,:,0] = c[0,0]*oimg[:,:,0] + c[0,1]*oimg[:,:,1] + c[0,2]*oimg[:,:,2]
	img[:,:,1] = c[1,0]*oimg[:,:,0] + c[1,1]*oimg[:,:,1] + c[1,2]*oimg[:,:,2]
	img[:,:,2] = c[2,0]*oimg[:,:,0] + c[2,1]*oimg[:,:,1] + c[2,2]*oimg[:,:,2]

	return img

def convFrom(img):
	a = matrix(([1/sqrt(3), 0, 0], [0, 1/sqrt(6), 0], [0, 0, 1/sqrt(2)]))
	b = matrix(([1, 1, 1], [1, 1, -2], [1, -1, 0])).astype(float)
	a = a.I
	b = b.I
	c = dot(b, a)

	oimg = img.copy()
	img[:,:,0] = c[0,0]*oimg[:,:,0] + c[0,1]*oimg[:,:,1] + c[0,2]*oimg[:,:,2]
	img[:,:,1] = c[1,0]*oimg[:,:,0] + c[1,1]*oimg[:,:,1] + c[1,2]*oimg[:,:,2]
	img[:,:,2] = c[2,0]*oimg[:,:,0] + c[2,1]*oimg[:,:,1] + c[2,2]*oimg[:,:,2]

	img[:,:,0] = exp(img[:,:,0])
	img[:,:,1] = exp(img[:,:,1])
	img[:,:,2] = exp(img[:,:,2])

	l = matrix(([0.3811, 0.5783, 0.0402], \
		[0.1967, 0.7244, 0.0782], \
		[0.0241, 0.1288, 0.8444]))
	l = l.I

	oimg = img.copy()
	img[:,:,0] = l[0,0]*oimg[:,:,0] + l[0,1]*oimg[:,:,1] + l[0,2]*oimg[:,:,2]
	img[:,:,1] = l[1,0]*oimg[:,:,0] + l[1,1]*oimg[:,:,1] + l[1,2]*oimg[:,:,2]
	img[:,:,2] = l[2,0]*oimg[:,:,0] + l[2,1]*oimg[:,:,1] + l[2,2]*oimg[:,:,2]		

	return img

def M(imgc):
	return imgc.sum()/imgc.shape[0]/imgc.shape[1]

def D(imgc):
	m = M(imgc)
	return sqrt(((imgc-m)**2).sum()/imgc.shape[0]/imgc.shape[1])

def statcorr(src, dst):
	print 'start'

	src = src.astype(float)
	dst = dst.astype(float)

	src = convTo(src)
	dst = convTo(dst)

	Ir = M(src[:,:,0]) + (dst[:,:,0] - M(dst[:,:,0]))*D(src[:,:,0])/D(dst[:,:,0])
	Ig = M(src[:,:,1]) + (dst[:,:,1] - M(dst[:,:,1]))*D(src[:,:,1])/D(dst[:,:,1])
	Ib = M(src[:,:,2]) + (dst[:,:,2] - M(dst[:,:,2]))*D(src[:,:,2])/D(dst[:,:,2])
	I = dstack((Ir,Ig,Ib))

	dst = convFrom(I)

	tmp = ones((dst.shape[0], dst.shape[1], dst.shape[2]), dtype = float) * 255

	tmp = (dst > 255) * tmp
	dst = ((dst >= 0)*(dst <= 255)*dst + tmp).astype(uint8)

	print 'end'

	return dst
