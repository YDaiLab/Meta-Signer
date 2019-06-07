# Third-party libraries
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC	
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, label_binarize
from sklearn.metrics import roc_curve
from utils.popphy_io import get_stat_dict

def train(train, test, config, metric, seed=42, max_iter=-1):

	num_cv = int(config.get('SVM', 'GridCV'))

	train_x, train_y = train
	test_x, test_y = test
	cl =  np.unique(train_y)
	num_class = len(cl)
	
	if num_class > 2 or metric == "AUC":
		scoring = "roc_auc"
	else:
		scoring = "accuracy"

	if num_class > 2:
		train_y = label_binarize(train_y, classes=cl)
		test_y = label_binarize(test_y, classes=cl)
	
		grid = [{'estimator__C': [1, 10, 100, 1000], 'estimator__kernel': ['linear']}]
		clf = GridSearchCV(OneVsRestClassifier(SVC(probability=True, max_iter=max_iter)), grid, cv=num_cv, scoring=scoring, n_jobs=-1, error_score='raise')
	
	else:	
		
		grid = [ {'C': [1, 10, 100, 1000], 'kernel': ['linear']}]
		clf = GridSearchCV(SVC(probability=True, max_iter=max_iter), grid, cv=num_cv, scoring=scoring, n_jobs=-1, error_score='raise')

	
	clf.fit(train_x, train_y)
	
	test_probs = np.array([row for row in clf.predict_proba(test_x)])
	test_pred = np.argmax(test_probs, axis=-1)

	test_stat_dict = get_stat_dict(test_y, test_probs)
	if num_class == 2:
		weights = np.array(clf.best_estimator_.coef_).reshape(-1)
		fpr, tpr, thresh = roc_curve(test_y, test_probs[:,1])
	else:
		weights = None
		fpr, tpr, thresh = None, None, None
	
	return test_stat_dict, tpr, fpr, thresh, weights, test_probs
