import sys
import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

from os.path import abspath
import numpy as np
import pandas as pd
import pickle

from utils.metasigner_io import get_config, load_params, get_stat_dict
from utils.generate_html import generate_html_feature_lists

import models.rf as rf
import models.svm as svm
import models.logistic_regression as logistic_regression
from models.mlpnn import MLPNN
from utils.tune import tune_mlpnn

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import argparse
import webbrowser
import subprocess

warnings.filterwarnings("ignore")

if __name__ == "__main__":

    #####################################################################
    # Read in Config File
    #####################################################################
    config = get_config()
    normalization = config.get('Evaluation', 'Normalization')
 
    parser = argparse.ArgumentParser(description='Meta-Signer Final Model Building')    
    parser.add_argument('ResultDir', metavar='--d',
                    help='Directory where Feature Rankings are Stored')
    parser.add_argument('NumFeatures', metavar='--k', help="Number of top features to select")
    parser.add_argument('-externalDir', '--e', dest="externalDir", nargs='?', help="External dataset to evaluate")
    args = parser.parse_args()

    dataset = args.ResultDir
    num_feat = int(args.NumFeatures)
    external_path = args.externalDir
    
    param_tune = True
    #########################################################################
    # Read in data and generate tree maps
    #########################################################################
    print("\nGenerating final models for %s using %d features..." % (dataset, num_feat))

    path = "../results/" + dataset
    abundance = pd.read_csv(path + "/abundance.tsv", sep="\t", index_col=0, header=0)
    permanova = pd.read_csv(path + "/PERMANOVA_rankings.csv", index_col=0)
    full_feature_set = abundance.index.values
    
    labels = pd.read_csv(path + "/labels.txt", index_col=0, header=None).index.values
    labels, label_set = pd.factorize(labels)    

    num_class = len(np.unique(labels))
        

    #########################################################################
    # Determine which models are being trained
    #########################################################################
  
    try:
        aggregated_rank_list = pd.read_csv(path + "/feature_evaluation/aggregated_rank_table.csv", header=None)
        ensemble_rank_list = pd.read_csv(path + "/feature_evaluation/ensemble_rank_table.csv", index_col=0, header=0)

    except:
        print("Warning: Could not open aggregated rank list! Exiting...")
        sys.exit()
        
    to_train = []
    
    try:
        rf_rankings = pd.read_csv(path + "/feature_evaluation/rf_feature_ranks.csv", index_col=0, header=0)
        rf_scores = pd.read_csv(path + "/feature_evaluation/rf_feature_scores.csv", index_col=0, header=0)
        rf_rank_list = pd.read_csv(path + "/feature_evaluation/rf_rank_list.txt").values
        train_rf = "True"
        to_train.append("RF")
    except:
        train_rf = "False"
        print("Warning: Could not find RF feature scores...")
     
    try:
        svm_rankings = pd.read_csv(path + "/feature_evaluation/svm_feature_ranks.csv", index_col=0, header=0)
        svm_rank_list = pd.read_csv(path + "/feature_evaluation/svm_rank_list.txt").values

        if num_class == 2:
            svm_scores = pd.read_csv(path + "/feature_evaluation/svm_feature_scores.csv", index_col=0, header=0)
        else:
            svm_scores = {}
            for l in label_set:
                svm_scores[l] = pd.read_csv(path + "/feature_evaluation/svm_feature_scores_" + str(l) + ".csv", index_col=0, header=0)
        train_svm = "True"
        to_train.append("SVM")
    except:
        train_svm = "False"
        print("Warning: Could not find SVM feature scores...")
        
    try:
        logistic_regression_rankings = pd.read_csv(path + "/feature_evaluation/logistic_regression_feature_ranks.csv", index_col=0, header=0)
        logistic_regression_rank_list = pd.read_csv(path + "/feature_evaluation/logistic_regression_rank_list.txt").values

        if num_class == 2:
            logistic_regression_scores = pd.read_csv(path + "/feature_evaluation/logistic_regression_feature_scores.csv", index_col=0, header=0)
        else:
            logistic_regression_scores = {}
            for l in label_set:
                logistic_regression_scores[l] = pd.read_csv(path + "/feature_evaluation/logistic_regression_feature_scores_" + str(l) + ".csv", index_col=0, header=0)
        train_logistic_regression = "True"
        to_train.append("Logistic Regression")
    except:
        train_logistic_regression = "False"
        print("Warning: Could not find Logistic Regression feature scores...")
        
    try:
        mlpnn_rankings = pd.read_csv(path + "/feature_evaluation/mlpnn_feature_ranks.csv", index_col=0, header=0)
        mlpnn_rank_list = pd.read_csv(path + "/feature_evaluation/mlpnn_rank_list.txt").values
        param_dict = load_params("../data/" + dataset)
        mlpnn_scores = {}
        for l in label_set:
            mlpnn_scores[l] = pd.read_csv(path + "/feature_evaluation/mlpnn_feature_scores_" + str(l) + ".csv", index_col=0, header=0)
        train_mlpnn = "True"
        to_train.append("MLPNN")
    except:
        train_mlpnn = "False"
        print("Warning: Could not find MLPNN feature scores...")
    rank_dict = {}
    features_subset = aggregated_rank_list.head(num_feat).values
    
    for feat in features_subset.reshape(-1):
        rank_dict[feat] = {}
        count = np.count_nonzero(ensemble_rank_list.head(int(num_feat)).values == feat)
        rank_dict[feat]["percent"] = str(np.round(float(count) / float(len(ensemble_rank_list.columns)),2))
        rank_dict[feat]["PERMANOVA_p"] = '{:.3e}'.format(permanova.loc[feat].values[0])
        rank_dict[feat]["PERMANOVA_rank"] = str(permanova.sort_values(by="Adj p-value", ascending=True).index.get_loc(feat) + 1)
        enriched_lab="-"

        votes = np.zeros(num_class)


        if "SVM" in to_train: 
            if num_class == 2:
                if np.median(svm_scores.loc[feat].values) > np.quantile(svm_scores.loc[feat].values, 0.75):
                    votes[1] += 1
                elif np.median(svm_scores.loc[feat].values) < np.quantile(svm_scores.loc[feat].values, 0.25):
                    votes[0] += 1
            else:
                max_val = 0
                enriched_lab = "-"
                for lab in range(len(label_set)):
                    if np.median(svm_scores[label_set[lab]].loc[feat].values) > max_val:
                        max_value = np.median(svm_scores[label_set[lab]].loc[feat].values)
                        max_lab = lab
                votes[max_lab] += 1

        if "Logistic Regression" in to_train:
            if num_class == 2:
                if np.median(logistic_regression_scores.loc[feat].values) > np.quantile(logistic_regression_scores.loc[feat].values, 0.75):
                    votes[1] += 1
                elif np.median(logistic_regression_scores.loc[feat].values) < np.quantile(logistic_regression_scores.loc[feat].values, 0.25):
                    votes[0] += 1
            else:
                max_val = 0
                enriched_lab = "-"
                for lab in range(len(label_set)):
                    if np.median(logistic_regression_scores[label_set[lab]].loc[feat].values) > max_val:
                        max_value = np.median(logistic_regression_scores[label_set[lab]].loc[feat].values)
                        max_lab = lab
                votes[max_lab] += 1

        if "MLPNN" in to_train:
            if num_class == 2:
                if np.median(mlpnn_scores[label_set[0]].loc[feat].values) > np.median(mlpnn_scores[label_set[1]].loc[feat].values):
                    votes[1] += 1
                else:
                    votes[0] += 1
            else:
                max_val = 0
                enriched_lab = "-"
                for lab in range(len(label_set)):
                    if np.median(mlpnn_scores[label_set[lab]].loc[feat].values) > max_val:
                        max_value = np.median(mlpnn_scores[label_set[lab]].loc[feat].values)
                        max_lab = lab
                votes[max_lab] += 1

        rank_dict[feat]["Enriched"] = label_set[np.argmax(votes)]



        model_rankings = {}
        if "RF" in to_train:
            model_rankings["RF"] = {}
            model_rankings["RF"]["Features"] = rf_rank_list[:num_feat]
            model_rankings["RF"]["Percent"] = []
            for feat in model_rankings["RF"]["Features"]:
                count = np.count_nonzero(rf_rankings.head(int(num_feat)).values == feat)
                model_rankings["RF"]["Percent"].append(str(np.round(float(count) / float(len(rf_rankings.columns)) * 100,1)))            

        if "SVM" in to_train:
            model_rankings["SVM"] = {}
            model_rankings["SVM"]["Features"] = svm_rank_list[:num_feat]
            model_rankings["SVM"]["Percent"] = []
            for feat in model_rankings["SVM"]["Features"]:
                count = np.count_nonzero(svm_rankings.head(int(num_feat)).values == feat)
                model_rankings["SVM"]["Percent"].append(str(np.round(float(count) / float(len(svm_rankings.columns)) * 100,1)))            

        if "Logistic Regression":
            model_rankings["Logistic Regression"] = {}
            model_rankings["Logistic Regression"]["Features"] = logistic_regression_rank_list[:num_feat]
            model_rankings["Logistic Regression"]["Percent"] = []
            for feat in model_rankings["Logistic Regression"]["Features"]:
                count = np.count_nonzero(logistic_regression_rankings.head(int(num_feat)).values == feat)
                model_rankings["Logistic Regression"]["Percent"].append(str(np.round(float(count) / float(len(logistic_regression_rankings.columns)) * 100,1)))           

        if "MLPNN" in to_train:
            model_rankings["MLPNN"] = {}
            model_rankings["MLPNN"]["Features"] = mlpnn_rank_list[:num_feat]
            model_rankings["MLPNN"]["Percent"] = []
            for feat in model_rankings["MLPNN"]["Features"]:
                count = np.count_nonzero(mlpnn_rankings.head(int(num_feat)).values == feat)
                model_rankings["MLPNN"]["Percent"].append(str(np.round(float(count) / float(len(mlpnn_rankings.columns)) * 100,1)))            



                
    
    print("Evaluating Final Models...")
    abundance_sub = abundance.loc[np.intersect1d(abundance.index.values, features_subset.reshape(-1))].values

    n_values = np.max(labels) + 1
    labels_oh = np.eye(n_values)[labels]

    abundance_sub = np.transpose(abundance_sub)
    log_abundance_sub = np.log(abundance_sub + 1)
    
    results_sub_df = pd.DataFrame(index=["AUC", "MCC", "Precision", "Recall", "F1"], columns=["RF", "SVM", "Logistic Regression", "MLPNN"])
    auc_sub_list = {}
    mcc_sub_list = {}
    precision_sub_list = {}
    recall_sub_list = {}
    f1_sub_list = {}
    
    for m in results_sub_df.columns:
        auc_sub_list[m]=[]
        mcc_sub_list[m]=[]
        precision_sub_list[m]=[]
        recall_sub_list[m]=[]
        f1_sub_list[m]=[]

    try:
        os.mkdir(path + "/final_models_" + str(num_feat))
    except OSError:
        print("Warning: Creation of the final model subdirectory failed...")


    if normalization == "MinMax":
        scaler = MinMaxScaler().fit(log_abundance_sub)
        train_x = np.clip(scaler.transform(log_abundance_sub), 0, 1)

    if normalization == "Standard":
        scaler = StandardScaler().fit(log_abundance_sub)
        train_x = np.clip(scaler.transform(log_abundance_sub), -3, 3)

    train = (train_x, labels) 
    train_oh = (train_x, labels_oh)
    
    if train_rf == "True":
            
        rf_clf, rf_stats, _ = rf.train(train, train, config)        
            
        auc_sub_list["RF"].append(rf_stats["AUC"])
        mcc_sub_list["RF"].append(rf_stats["MCC"])
        precision_sub_list["RF"].append(rf_stats["Precision"])
        recall_sub_list["RF"].append(rf_stats["Recall"])
        f1_sub_list["RF"].append(rf_stats["F1"])
        
        with open(path + "/final_models_" + str(num_feat) + "/rf_model.pkl", "wb") as f:
            pickle.dump(rf_clf, f)

        results_sub_df.loc["AUC"]["RF"] = "{:.2f}".format(np.mean(rf_stats["AUC"]))
        results_sub_df.loc["MCC"]["RF"] = "{:.2f}".format(np.mean(rf_stats["MCC"])) 
        results_sub_df.loc["Precision"]["RF"] = "{:.2f}".format(np.mean(rf_stats["Precision"]))
        results_sub_df.loc["Recall"]["RF"] = "{:.2f}".format(np.mean(rf_stats["Recall"]))
        results_sub_df.loc["F1"]["RF"] = "{:.2f}".format(np.mean(rf_stats["F1"]))
        
    if train_svm == "True":

        svm_clf, svm_stats, _ = svm.train(train, train, config)

        auc_sub_list["SVM"].append(svm_stats["AUC"])
        mcc_sub_list["SVM"].append(svm_stats["MCC"])
        precision_sub_list["SVM"].append(svm_stats["Precision"])
        recall_sub_list["SVM"].append(svm_stats["Recall"])
        f1_sub_list["SVM"].append(svm_stats["F1"])
        with open(path + "/final_models_" + str(num_feat) + "/svm_model.pkl", "wb") as f:
            pickle.dump(svm_clf, f)
            
        results_sub_df.loc["AUC"]["SVM"] = "{:.2f}".format(np.mean(svm_stats["AUC"]))
        results_sub_df.loc["MCC"]["SVM"] = "{:.2f}".format(np.mean(svm_stats["MCC"]))
        results_sub_df.loc["Precision"]["SVM"] = "{:.2f}".format(np.mean(svm_stats["Precision"]))
        results_sub_df.loc["Recall"]["SVM"] = "{:.2f}".format(np.mean(svm_stats["Recall"]))
        results_sub_df.loc["F1"]["SVM"] = "{:.2f}".format(np.mean(svm_stats["F1"]))

    if train_logistic_regression == "True":
            
        lr_clf, lr_stats, _ = logistic_regression.train(train, train, config, regularization=False)

        auc_sub_list["Logistic Regression"].append(lr_stats["AUC"])
        mcc_sub_list["Logistic Regression"].append(lr_stats["MCC"])
        precision_sub_list["Logistic Regression"].append(lr_stats["Precision"])
        recall_sub_list["Logistic Regression"].append(lr_stats["Recall"])
        f1_sub_list["Logistic Regression"].append(lr_stats["F1"])                    
        with open(path + "/final_models_" + str(num_feat) + "/logistic_regression_model.pkl", "wb") as f:
            pickle.dump(lr_clf, f)

        results_sub_df.loc["AUC"]["Logistic Regression"] = "{:.2f}".format(np.mean(lr_stats["AUC"]))
        results_sub_df.loc["MCC"]["Logistic Regression"] = "{:.2f}".format(np.mean(lr_stats["MCC"]))
        results_sub_df.loc["Precision"]["Logistic Regression"] = "{:.2f}".format(np.mean(lr_stats["Precision"]))
        results_sub_df.loc["Recall"]["Logistic Regression"] = "{:.2f}".format(np.mean(lr_stats["Recall"]))
        results_sub_df.loc["F1"]["Logistic Regression"] = "{:.2f}".format(np.mean(lr_stats["F1"]))
    
    if train_mlpnn == "True":
            
        mlpnn_model = MLPNN(train_x.shape[1], num_class, config, param_dict["MLPNN"])
        mlpnn_model.train(train_oh)            
        mlpnn_stats = mlpnn_model.test(train_oh)

        auc_sub_list["MLPNN"].append(mlpnn_stats["AUC"])
        mcc_sub_list["MLPNN"].append(mlpnn_stats["MCC"])
        precision_sub_list["MLPNN"].append(mlpnn_stats["Precision"])
        recall_sub_list["MLPNN"].append(mlpnn_stats["Recall"])
        f1_sub_list["MLPNN"].append(mlpnn_stats["F1"])                                                           
        mlpnn_model.save(path + "/final_models_" + str(num_feat) + "/")      

        results_sub_df.loc["AUC"]["MLPNN"] = "{:.2f}".format(np.mean(mlpnn_stats["AUC"]))
        results_sub_df.loc["MCC"]["MLPNN"] = "{:.2f}".format(np.mean(mlpnn_stats["MCC"]))
        results_sub_df.loc["Precision"]["MLPNN"] = "{:.2f}".format(np.mean(mlpnn_stats["Precision"]))
        results_sub_df.loc["Recall"]["MLPNN"] = "{:.2f}".format(np.mean(mlpnn_stats["Recall"]))
        results_sub_df.loc["F1"]["MLPNN"] = "{:.2f}".format(np.mean(mlpnn_stats["F1"]))

    results_sub_df.to_csv(path + "/final_models_" + str(num_feat) + "/training_results.tsv", sep="\t")
    
    if external_path != None:

        external_data_df = pd.read_csv("../data/" + external_path + "/abundance.tsv", index_col=0, header=None, sep="\t").loc[np.intersect1d(abundance.index.values, features_subset.reshape(-1))]
        external_labels_df = pd.read_csv("../data/" + external_path + "/labels.txt", index_col=0, header=None).index.values
        results_external_sub_df = pd.DataFrame(index=["AUC", "MCC", "Precision", "Recall", "F1"], 
                                                       columns=["RF", "SVM", "Logistic Regression", "MLPNN"])
        
        log_external_sub = np.log(external_data_df.transpose().values + 1)

        if normalization == "MinMax":
            test = np.clip(scaler.transform(log_external_sub), 0, 1)
        if normalization == "Standard":
            test = np.clip(scaler.transform(log_external_sub), -3, 3)

        external_labels = []
        for lab in external_labels_df:
            external_labels.append(np.where(label_set==lab)[0])
        external_labels = np.array(external_labels).reshape(-1)
        external_labels_oh = np.eye(n_values)[external_labels]

        external_data = (test, external_labels)
        external_data_oh = (test, external_labels_oh)

        if train_rf == "True":
            pred = np.argmax(rf_clf.predict_proba(test), axis=1)
            probs = rf_clf.predict_proba(test)
            rf_stats = get_stat_dict(external_labels, probs, pred)
            results_external_sub_df.loc["AUC"]["RF"] = "{:.2f}".format(rf_stats["AUC"])
            results_external_sub_df.loc["MCC"]["RF"] = "{:.2f}".format(rf_stats["MCC"])
            results_external_sub_df.loc["Precision"]["RF"] = "{:.2f}".format(rf_stats["Precision"])
            results_external_sub_df.loc["Recall"]["RF"] = "{:.2f}".format(rf_stats["Recall"])
            results_external_sub_df.loc["F1"]["RF"] = "{:.2f}".format(rf_stats["F1"])

        if train_svm == "True":
            pred = np.argmax(svm_clf.predict_proba(test), axis=1)
            probs = svm_clf.predict_proba(test)
            svm_stats = get_stat_dict(external_labels, probs, pred)
            results_external_sub_df.loc["AUC"]["SVM"] = "{:.2f}".format(svm_stats["AUC"])
            results_external_sub_df.loc["MCC"]["SVM"] = "{:.2f}".format(svm_stats["MCC"])
            results_external_sub_df.loc["Precision"]["SVM"] = "{:.2f}".format(svm_stats["Precision"])
            results_external_sub_df.loc["Recall"]["SVM"] = "{:.2f}".format(svm_stats["Recall"])
            results_external_sub_df.loc["F1"]["SVM"] = "{:.2f}".format(svm_stats["F1"])
            
        if train_logistic_regression == "True":
            pred =  np.argmax(lr_clf.predict_proba(test), axis=1)
            probs = lr_clf.predict_proba(test)
            lr_stats = get_stat_dict(external_labels, probs, pred)
            results_external_sub_df.loc["AUC"]["Logistic Regression"] = "{:.2f}".format(lr_stats["AUC"])
            results_external_sub_df.loc["MCC"]["Logistic Regression"] = "{:.2f}".format(lr_stats["MCC"])
            results_external_sub_df.loc["Precision"]["Logistic Regression"] = "{:.2f}".format(lr_stats["Precision"])
            results_external_sub_df.loc["Recall"]["Logistic Regression"] = "{:.2f}".format(lr_stats["Recall"])
            results_external_sub_df.loc["F1"]["Logistic Regression"] = "{:.2f}".format(lr_stats["F1"])
            
        if train_mlpnn == "True":
            mlpnn_stats = mlpnn_model.test((test, external_labels_oh))
            results_external_sub_df.loc["AUC"]["MLPNN"] = "{:.2f}".format(mlpnn_stats["AUC"])
            results_external_sub_df.loc["MCC"]["MLPNN"] = "{:.2f}".format(mlpnn_stats["MCC"])
            results_external_sub_df.loc["Precision"]["MLPNN"] = "{:.2f}".format(mlpnn_stats["Precision"])
            results_external_sub_df.loc["Recall"]["MLPNN"] = "{:.2f}".format(mlpnn_stats["Recall"])
            results_external_sub_df.loc["F1"]["MLPNN"] = "{:.2f}".format(mlpnn_stats["F1"])
    
        results_external_sub_df.to_csv(path + "/final_models_" + str(num_feat) + "/external_results.tsv", sep="\t")
        print(results_external_sub_df)
        
        generate_html_feature_lists(dataset, path + "/final_models_" + str(num_feat), config, to_train, model_rankings, rank_dict, 
                                    results_sub_df, results_external_sub_df, label_set,has_external = True)  
    else:
        generate_html_feature_lists(dataset, path + "/final_models_" + str(num_feat), config, to_train, model_rankings, rank_dict, 
                                    results_sub_df, None, label_set)  
