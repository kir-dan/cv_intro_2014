#!/usr/bin/python

from sys import argv, stdout, exit
from numpy import array, loadtxt
import pandas as pd
from fit_and_classify import fit_and_classify

if len(argv) != 3:
    stdout.write('Usage: %s train_file test_file\n' % argv[0])
    exit(1)

train_file = argv[1]
test_file = argv[2]

train = loadtxt(train_file, delimiter=',', skiprows=1)
train_features = train[:, :-1]
train_labels = train[:, -1]

test = loadtxt(test_file, delimiter=',', skiprows=1)
test_features = test[:, :-1]
test_labels = test[:, -1]

stdout.write('%.4f\n' % (sum(test_labels == fit_and_classify(train_features, train_labels, test_features)) / float(test_labels.shape[0])))
