#!/usr/bin/python

from sys import argv, stdout, exit
from os.path import split, splitext
from skimage.io import imread, imsave
from statcorr import statcorr
from csv import reader

if len(argv) != 3:
    stdout.write('Usage: %s picture_list.txt output_dir\n' % argv[0])
    exit(1)

path, name = split(argv[1])
with open(argv[1], 'rb') as desc:
    strs = reader(desc, delimiter=' ')
    for line in strs:
        imsave(argv[2] + '/' + splitext(line[0])[0]           \
                       + '_' + splitext(line[1])[0] + '.png', \
               statcorr(imread(path + '/' + line[0]),         \
                        imread(path + '/' + line[1])))
