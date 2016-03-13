#!/usr/bin/python

from sys import argv, stdout, exit
from os.path import basename
from glob import iglob
from align import align
from skimage.io import imread, imsave

if len(argv) != 3:
    stdout.write('Usage: %s input_dir_path output_dir_path\n' % argv[0])
    exit(1)

input_dir_path = argv[1]
output_dir_path = argv[2]

for filename in iglob(input_dir_path + '/*.png'):
    img = imread(filename)
    img = align(img)
    img = img.astype('float64') / 255
    imsave(output_dir_path + '/' + basename(filename), img)
