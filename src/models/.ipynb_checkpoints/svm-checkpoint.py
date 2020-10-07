# Third-party libraries
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC	
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, label_binarize
from sklearn.metrics import roc_curve
from utils.popphy_io import get_stat_dict

def train(train, test, config, metric, seed=42, max_iter=100000, gaussian=False):

    num_cv = int(config.get('SVM', 'GridCV'))

    train_x, train_y = train
    test_x, test_y = test
    cl =  np.unique(train_y)
    num_class = len(cl)
    
    scoring = "roc_auc"
    
    if num_class > 2:
        train_y_binarize = label_binarize(train_y, classes=cl)
        test_y_binarize = label_binarize(test_y, classes=cl)
        
        if gaussian==True:
            grid = [{'estimator__kernel': ['rbf'], 'estimator__gamma': [1e-3, 1e-4], 'estimator__C': [1, 10, 100, 1000]},
                          {'estimator__kernel': ['linear'], 'estimator__C': [1, 10, 100, 1000]}]
        else:
            grid = [{'estimator__C': [1, 10, 100, 1000], 'estimator__kernel': ['linear']}]
            
        clf = GridSearchCV(OneVsRestClassifier(SVC(probability=True, max_iter=max_iter)),
                               grid, cv=StratifiedKFold(num_cv).split(train_x, train_y), 
                               scoring=scoring, n_jobs=-1)
        clf.fit(train_x, train_y_binarize)

    else:
        if gaussian==True:
            grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                          {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}] 
        else:
            grid = [ {'C': [1, 10, 100, 1000], 'kernel': ['linear']}]
        clf = GridSearchCV(SVC(probability=True, max_iter=max_iter), 
                               grid, cv=StratifiedKFold(num_cv).split(train_x, train_y), scoring=scoring, n_jobs=-1)


        clf.fit(train_x, train_y)

    test_probs = clf.predict_proba(test_x)
    test_preds = np.argmax(clf.predict_proba(test_x), axis=1)

    test_stat_dict = get_stat_dict(test_y, test_probs, test_preds)
    
    if num_class == 2 and gaussian == False:
        weights = np.array(clf.best_estimator_.coef_).reshape(-1)
        fpr, tpr, thresh = roc_curve(test_y, test_probs[:,1])
    elif num_class > 2 and gaussian == False:
        weights = np.array(clf.best_estimator_.coef_)
        fpr, tpr, thresh = None, None, None
    elif gaussian == True:
        weights = None
        fpr, tpr, thresh = None, None, None
    
    return clf.best_estimator_, test_stat_dict, tpr, fpr, thresh, weights, test_probs
