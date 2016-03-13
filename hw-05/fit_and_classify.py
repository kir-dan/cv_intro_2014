# python2
from numpy import ones
from sklearn import neighbors, cross_validation, svm

def fit_and_classify(train_features, train_labels, test_features):
#	X_train, X_test, y_train, y_test = cross_validation.train_test_split( \
#		train_features, train_labels, test_size=0.2, random_state=0)


	clf = svm.SVC(kernel='rbf', C=35, gamma=7).fit(train_features, train_labels)
#	print clf.score(X_test, y_test)

	return clf.predict(test_features)