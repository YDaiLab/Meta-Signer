# Third-party libraries
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC	
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from utils.metasigner_io import get_stat_dict

def train(train, test, config, seed=42, gaussian=False):

    num_cv = int(config.get('SVM', 'GridCV'))
    max_iter = int(config.get('SVM', 'MaxIterations'))

    train_x, train_y = train
    test_x, test_y = test
    cl =  np.unique(train_y)
    num_class = len(cl)
  
    
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
                               scoring="roc_auc", n_jobs=-1)
        clf.fit(train_x, train_y_binarize)

    else:
        if gaussian==True:
            grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                          {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}] 
        else:
            grid = [ {'C': [1, 10, 100, 1000], 'kernel': ['linear']}]

        clf = GridSearchCV(SVC(probability=True, max_iter=max_iter), 
                               grid, cv=StratifiedKFold(num_cv).split(train_x, train_y), scoring="roc_auc", n_jobs=-1)


        clf.fit(train_x, train_y)

    test_probs = clf.predict_proba(test_x)
    test_preds = np.argmax(clf.predict_proba(test_x), axis=1)

    test_stat_dict = get_stat_dict(test_y, test_probs, test_preds)
    
    if num_class == 2 and gaussian == False:
        weights = np.array(clf.best_estimator_.coef_).reshape(-1)
    elif num_class > 2 and gaussian == False:
        weights = np.array(clf.best_estimator_.coef_)
    elif gaussian == True:
        weights = None

    
    return clf.best_estimator_, test_stat_dict, weights
