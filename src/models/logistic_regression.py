# Third-party libraries
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from utils.metasigner_io import get_stat_dict


def train(train, test, config, seed=42, regularization = True):

    n_iter = int(config.get('Logistic Regression', 'MaxIterations'))
    num_cv = int(config.get('Logistic Regression', 'GridCV'))

    train_x, train_y = train
    test_x, test_y = test
    cl = np.unique(train_y)
    num_class = len(cl)

    if regularization:
        grid={"C":np.logspace(-3,3,50), "penalty":["l1"]}
        clf = GridSearchCV(LogisticRegression(solver="saga", max_iter=n_iter), grid, cv=StratifiedKFold(num_cv).split(train_x, train_y))
        clf.fit(train_x,train_y)
        clf = clf.best_estimator_
    else:
        clf = LogisticRegression(solver="saga", max_iter=n_iter)
        clf.fit(train_x,train_y)
        

    test_probs = clf.predict_proba(test_x)
    test_preds = np.argmax(clf.predict_proba(test_x), axis=1)
    
    test_stat_dict = get_stat_dict(test_y, test_probs, test_preds)


    weights = np.array(clf.coef_)
            

    return clf, test_stat_dict, weights
