# python2
from sklearn.svm import LinearSVC
from datetime import *

def fit_and_classify(train_features, train_labels, test_features):

	print "Let's learn!"
	print datetime.now()

	clf = LinearSVC(C = 1).fit(train_features, train_labels)

	print "Let's predict!"
	print datetime.now()

	return clf.predict(test_features)
