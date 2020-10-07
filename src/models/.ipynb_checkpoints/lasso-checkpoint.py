# Third-party libraries
import numpy as np
from sklearn.linear_model import LassoCV, Lasso
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, label_binarize
from utils.popphy_io import get_stat_dict
from sklearn.metrics import roc_curve

def train(train, test, config, metric, seed=42):

	n_iter = int(config.get('LASSO', 'NumberIterations'))
	num_cv = int(config.get('LASSO', 'GridCV'))

	train_x, train_y = train
	test_x, test_y = test
	cl = np.unique(train_y)
	num_class = len(cl)

	if num_class > 2:
		train_y = label_binarize(train_y, classes=cl)
		test_y = label_binarize(test_y, classes=cl)
		clf = OneVsRestClassifier(LassoCV(alphas=np.logspace(-4, -0.5, 50), cv=num_cv, n_jobs=-1, max_iter=n_iter))
	else:
		clf = LassoCV(alphas=np.logspace(-4, -0.5, 50), cv=num_cv, n_jobs=-1, max_iter=n_iter)
	clf.fit(train_x, train_y)

	if num_class == 2:
		test_probs = np.array([[1-row, row] for row in clf.predict(test_x)])
		test_pred = np.argmax(test_probs, axis=-1)
		test_stat_dict = get_stat_dict(test_y, test_probs)
		fpr, tpr, thresh = roc_curve(test_y, test_probs[:,1])
		weights = clf.coef_
	else:
		test_pred = clf.predict(test_x)
		test_probs = clf.predict(test_x)
		test_stat_dict = get_stat_dict(test_y, test_pred)
		fpr, tpr, thresh = None, None, None
		weights = None

	return clf, test_stat_dict, tpr, fpr, thresh, weights, test_probs
