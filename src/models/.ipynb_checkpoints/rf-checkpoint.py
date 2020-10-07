# Third-party libraries
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold
from utils.popphy_io import get_stat_dict, get_stat
from sklearn.metrics import roc_curve

def train(train, test, config, metric, seed=42, feature_select=True):

    number_trees = int(config.get('RF', 'NumberTrees'))
    num_models = int(config.get('RF', 'ValidationModels'))

    x, y = train
    test_x, test_y = test

    if metric == "AUC":
        scoring = "roc_auc"
    else:
        scoring = "accuracy"
    
    clf = RandomForestClassifier(n_estimators=number_trees, n_jobs=-1)
    clf.fit(x, y)
    
    feature_importance = clf.feature_importances_
    
    feature_ranking = np.flip(np.argsort(feature_importance))
    num_features = x.shape[1]
    best_num_features = num_features

    if feature_select:
        percent_features = [1.0, 0.75, 0.5, 0.25]

        skf = StratifiedKFold(n_splits=num_models, shuffle=True)

        best_score = -1
    
        for percent in percent_features:
            run_score = -1
            run_probs = []
            for train_index, valid_index in skf.split(x, y):
                train_x, valid_x = x[train_index], x[valid_index]
                train_y, valid_y = y[train_index], y[valid_index]
                
                features_using = int(round(num_features * percent))
                feature_list = feature_ranking[0:features_using]
                filtered_train_x = train_x[:,feature_list]
                filtered_valid_x = valid_x[:,feature_list]
                clf = RandomForestClassifier(n_estimators=number_trees, n_jobs=-1).fit(filtered_train_x, train_y)
                probs = [row for row in clf.predict_proba(filtered_valid_x)]
                run_probs = list(run_probs) + list(probs)
            run_score = get_stat(y, run_probs, metric)

            if run_score > best_score:
                best_num_features = num_features


    feature_list = feature_ranking[0:best_num_features]
    x_filt = x[:,feature_list]
    test_x_filt = test_x[:,feature_list]


    clf = RandomForestClassifier(n_estimators=number_trees, n_jobs=-1).fit(x, y)

    test_probs = np.array([row for row in clf.predict_proba(test_x)])
    test_pred = np.argmax(test_probs, axis=-1)

    test_stat_dict = get_stat_dict(test_y, test_probs, test_pred)
    
    if len(np.unique(y)) == 2:
        fpr, tpr, thresh = roc_curve(test_y, test_probs[:,1])
    else:
        fpr, tpr, thresh = 0, 0, 0
    return clf, test_stat_dict, tpr, fpr, thresh, feature_importance, test_probs