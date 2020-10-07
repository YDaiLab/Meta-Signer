import sys
import os
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from os.path import abspath
import numpy as np
import pandas as pd

from utils.metasigner_io import get_config, save_params, load_params, get_stat_dict
from utils.generate_html import generate_html_performance
from utils.generate_plots import generate_boxplot, generate_score_distributions, generate_feature_selection_curve
from utils.permanova_test import get_permanova_ranked_list

import models.rf as rf
import models.svm as svm
import models.logistic_regression as logistic_regression
from models.mlpnn import MLPNN
from utils.tune import tune_mlpnn

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import webbrowser
import subprocess

warnings.filterwarnings("ignore")

if __name__ == "__main__":

    #####################################################################
    # Read in Config File
    #####################################################################

    config = get_config()
    datasets = config.get('Evaluation', 'DataSet').split(",")
    num_runs = int(config.get('Evaluation', 'NumberRuns'))
    num_test = int(config.get('Evaluation', 'NumberTestSplits'))
    train_rf = config.get('RF', 'Train')
    train_svm = config.get('SVM', 'Train')
    train_logistic_regression = config.get('Logistic Regression', 'Train')
    train_mlpnn = config.get('MLPNN', 'Train')
    filt_thresh_count = config.get('Evaluation', 'FilterThreshCount')
    filt_thresh_mean = config.get('Evaluation', 'FilterThreshMean')
    normalization = config.get('Evaluation', 'Normalization')
    max_k = config.get('Evaluation', 'MaxK')
    agg_method = config.get('Evaluation', 'AggregateMethod')
    

    if agg_method != "GA" and agg_method != "CE":
        agg_method = "CE"
        
    for dataset in datasets:

        param_tune = True
        #########################################################################
        # Read in data and generate tree maps
        #########################################################################
        print("\nStarting Meta-Signer on %s..." % (dataset))

        path = "../data/" + dataset
        abundance = pd.read_csv(path + "/abundance.tsv", sep="\t", index_col=0, header=None)
        labels = pd.read_csv(path + "/labels.txt", index_col=0, header=None)
        labels_data, label_set = pd.factorize(labels.index.values)    
        num_class = len(np.unique(labels_data))
        full_features = abundance.index
        permanova_df = get_permanova_ranked_list(abundance, labels_data, full_features, label_set)
        
        col_sums = abundance.sum(axis=0)
        abundance = abundance.divide(col_sums, axis=1)
        num_pos = (abundance != 0).astype(int).sum(axis=1)
        
        abundance = abundance.drop(num_pos.loc[num_pos.values < float(filt_thresh_count) * abundance.values.shape[1]].index)
        abundance = abundance.loc[abundance.mean(1) > float(filt_thresh_mean)]        
        features = abundance.index.values
        
        #########################################################################
        # Determine which models are being trained
        #########################################################################

        to_train = []

        if train_rf == "True":
            to_train.append("RF")


        if train_svm == "True":
            to_train.append("SVM")


        if train_logistic_regression == "True":
            to_train.append("Logistic Regression")


        if train_mlpnn == "True":
            to_train.append("MLPNN")

        #########################################################################
        # Set up DataFrames to store results
        #########################################################################

        cv_list = ["Run_" + str(x) + "_CV_" + str(y) for x in range(num_runs) for y in range(num_test)]

        auc_df = pd.DataFrame(index=to_train, columns=cv_list)
        mcc_df = pd.DataFrame(index=to_train, columns=cv_list)
        precision_df = pd.DataFrame(index=to_train, columns=cv_list)
        recall_df = pd.DataFrame(index=to_train, columns=cv_list)
        f1_df = pd.DataFrame(index=to_train, columns=cv_list)

        #########################################################################
        # Create results directory for current run
        #########################################################################

        result_path = "../results/" + dataset
        print("Saving results to %s" % (result_path))
        try:
            os.mkdir("../results/")
        except OSError:
            pass

        try:
            os.mkdir(result_path)
        except OSError:
            print("Warning: result subdirectory already exists...")

        try:
            os.mkdir(result_path + "/prediction_evaluation")
        except OSError:
            pass

        try:
            os.mkdir(result_path + "/feature_evaluation")
        except OSError:
            pass
        
        labels.to_csv(result_path + "/labels.txt", header=None)


        print("There are %d classes...%s" % (num_class, ", ".join(label_set)))
        
        permanova_df.to_csv(result_path + "/PERMANOVA_rankings.csv")
        
        np.savetxt(result_path + "/label_set.txt", label_set, fmt="%s")
        abundance.to_csv(result_path + "/abundance.tsv", sep="\t")

        abundance = abundance.transpose().values        
        labels = labels_data

        seed = np.random.randint(100)
        np.random.seed(seed)
        np.random.shuffle(abundance)
        np.random.seed(seed)
        np.random.shuffle(labels)
        
        n_values = np.max(labels) + 1
        labels_oh = np.eye(n_values)[labels]

        #########################################################################
        # Set up seeds for different runs
        #########################################################################

        rf_scores = pd.DataFrame(index=features)
        
        if num_class == 2:
            svm_scores = pd.DataFrame(index=features)
            logistic_regression_scores = pd.DataFrame(index=features)

        else:
            svm_scores = {}
            logistic_regression_scores = {}
            for l in label_set:
                svm_scores[l] = pd.DataFrame(index=features)     
                logistic_regression_scores[l] = pd.DataFrame(index=features) 
                
        mlpnn_scores = {}
        for l in label_set:
            mlpnn_scores[l] = pd.DataFrame(index=features)

        seeds = np.random.randint(1000, size=num_runs)
        run = 0

        for seed in seeds:

            #####################################################################
            # Set up CV partitioning
            #####################################################################

            print("Starting CV")
            skf = StratifiedKFold(n_splits=num_test, shuffle=True, random_state=seed)
            fold = 0

            #####################################################################
            # Begin k-fold CV
            #####################################################################
            for train_index, test_index in skf.split(abundance, labels):

                #################################################################
                # Select and format training and testing sets
                #################################################################
                train_x, test_x = abundance[train_index,:], abundance[test_index,:]
                train_y, test_y = labels[train_index], labels[test_index]
                train_y_oh, test_y_oh = labels_oh[train_index,:], labels_oh[test_index,:]
                
                train_x = np.log(train_x + 1)
                test_x = np.log(test_x + 1)

                num_train_samples = train_x.shape[0]
                num_test_samples = test_x.shape[0]
                
                if normalization == "MinMax":
                    scaler = MinMaxScaler().fit(train_x)
                    train_x = np.clip(scaler.transform(train_x), 0, 1)
                    test_x = np.clip(scaler.transform(test_x), 0, 1)

                if normalization == "Standard":
                    scaler = StandardScaler().fit(train_x)
                    train_x = np.clip(scaler.transform(train_x), -3, 3)
                    test_x = np.clip(scaler.transform(test_x), -3, 3)

                
                train_x = np.array(train_x)
                train_y = np.array(train_y)

                train = [train_x, train_y]
                test = [test_x, test_y]
                
                train_mlpnn_raw = [train_x, train_y_oh]
                test_mlpnn_raw = [test_x, test_y_oh]

                #################################################################
                # Try to load model parameters if first fold
                #################################################################

                if param_tune:
                    param_tune = False

                    try:
                        print("Loading Network Model Parameters...")
                        param_dict = load_params(path)

                    except:
                        print("Network parameter file not found...")
                        param_dict = {}

                    #############################################################
                    # Tune any parameters that are missing
                    #############################################################

                    if "MLPNN" not in param_dict and "MLPNN" in to_train:
                        print("MLPNN parameters not found...Tuning MLPNN paramters...")
                        param_dict["MLPNN"] = {}
                        mlpnn_layers, mlpnn_nodes, mlpnn_lamb, mlpnn_drop = tune_mlpnn(train_mlpnn_raw, 
                                                                                       test_mlpnn_raw, config)
                        param_dict["MLPNN"]["Num_Layers"] = mlpnn_layers
                        param_dict["MLPNN"]["Num_Nodes"] = mlpnn_nodes
                        param_dict["MLPNN"]["L2_Lambda"] = mlpnn_lamb
                        param_dict["MLPNN"]["Dropout_Rate"] = mlpnn_drop

                    #############################################################
                    # Save parameters
                    #############################################################

                    save_params(param_dict, path)


                print("\n# AUC values for run %d fold %d:" % (run, fold))
                print("# Model\t\t\tAUC\t\tMean")

                #################################################################
                # Triain RF model using raw and tree features
                #################################################################

                if train_rf == "True":

                    _, stats, scores = rf.train(train, test, config)

                    rf_scores["Run_" + str(run) + "_CV_" + str(fold)] = scores

                    auc_df.loc["RF"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["AUC"]
                    mcc_df.loc["RF"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["MCC"]
                    precision_df.loc["RF"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["Precision"]
                    recall_df.loc["RF"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["Recall"]
                    f1_df.loc["RF"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["F1"]

                    sys.stdout.write("")
                    print("# RF:\t\t\t%f\t%f" % (stats["AUC"], auc_df.loc["RF"].mean(axis=0)))


                #################################################################
                # Triain SVM model using raw and tree features
                #################################################################

                if train_svm == "True":
                    _, stats, scores = svm.train(train, test, config)

                    if num_class == 2:
                        svm_scores["Run_" + str(run) + "_CV_" + str(fold)] = scores
                    else:
                        for l in range(len(label_set)):
                            score_list = scores[l,:].reshape(-1)
                            lab = label_set[l]
                            svm_scores[lab]["Run_" + str(run) + "_CV_" + str(fold)] = score_list
                            
                    auc_df.loc["SVM"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["AUC"]
                    mcc_df.loc["SVM"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["MCC"]
                    precision_df.loc["SVM"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["Precision"]
                    recall_df.loc["SVM"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["Recall"]
                    f1_df.loc["SVM"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["F1"]

                    sys.stdout.write("")
                    print("# SVM:\t\t\t%f\t%f" % (stats["AUC"], auc_df.loc["SVM"].mean(axis=0)))

                #################################################################
                # Triain Logistic Regression model using raw and tree features
                #################################################################

                if train_logistic_regression == "True":
                    _, stats, scores = logistic_regression.train(train, test, config)

                    if num_class == 2:
                        logistic_regression_scores["Run_" + str(run) + "_CV_" + str(fold)] = scores.reshape(-1)
                    else:
                        for l in range(len(label_set)):
                            score_list = scores[l,:].reshape(-1)
                            lab = label_set[l]
                            logistic_regression_scores[lab]["Run_" + str(run) + "_CV_" + str(fold)] = score_list

                    auc_df.loc["Logistic Regression"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["AUC"]
                    mcc_df.loc["Logistic Regression"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["MCC"]
                    precision_df.loc["Logistic Regression"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["Precision"]
                    recall_df.loc["Logistic Regression"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["Recall"]
                    f1_df.loc["Logistic Regression"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["F1"]

                    sys.stdout.write("")
                    print("# Logistic Regression:\t\t%f\t%f" % (stats["AUC"], 
                                                                   auc_df.loc["Logistic Regression"].mean(axis=0)))

                #################################################################
                # Triain MLPNN model using raw and tree features
                #################################################################
                if train_mlpnn == "True":
                    mlpnn_model = MLPNN(len(features), num_class, config, param_dict["MLPNN"])
                    mlpnn_model.train(train_mlpnn_raw)
                    stats = mlpnn_model.test(test_mlpnn_raw)
                    scores = mlpnn_model.get_scores()

                    auc_df.loc["MLPNN"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["AUC"]
                    mcc_df.loc["MLPNN"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["MCC"]
                    precision_df.loc["MLPNN"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["Precision"]
                    recall_df.loc["MLPNN"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["Recall"]
                    f1_df.loc["MLPNN"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["F1"]

                    sys.stdout.write("")
                    print("# MLPNN:\t\t%f\t%f" % (stats["AUC"], auc_df.loc["MLPNN"].mean(axis=0)))

                    for l in range(len(label_set)):
                        score_list = scores[:,l]
                        lab = label_set[l]
                        mlpnn_scores[lab]["Run_" + str(run) + "_CV_" + str(fold)] = score_list
                fold +=1
            run += 1
            
            #####################################################################
            # Save metric dataframes as files
            #####################################################################

        auc_df.to_csv(result_path + "/prediction_evaluation/results_AUC.tsv", sep="\t")
        mcc_df.to_csv(result_path + "/prediction_evaluation/results_MCC.tsv", sep="\t")
        precision_df.to_csv(result_path + "/prediction_evaluation/results_precision.tsv", sep="\t")
        recall_df.to_csv(result_path + "/prediction_evaluation/results_recall.tsv", sep="\t")
        f1_df.to_csv(result_path + "/prediction_evaluation/results_F1.tsv", sep="\t")

        #####################################################################
        # Store mean and std of metrics as file
        #####################################################################

        results_df = pd.DataFrame(index=["AUC", "MCC", "Precision", "Recall", "F1"], columns=to_train)

        
        for model in to_train:
            results_df.loc["AUC"][model] = "{:.2f}".format(auc_df.loc[model].mean())+ " (" + "{:.2f}".format(auc_df.loc[model].std()) + ")"                                                        
            results_df.loc["MCC"][model] = "{:.2f}".format(mcc_df.loc[model].mean()) + " (" + "{:.2f}".format(mcc_df.loc[model].std()) + ")"
            results_df.loc["Precision"][model] = "{:.2f}".format(precision_df.loc[model].mean()) + " (" + "{:.2f}".format(precision_df.loc[model].std()) + ")"
            results_df.loc["Recall"][model] = "{:.2f}".format(recall_df.loc[model].mean()) + " (" + "{:.2f}".format(recall_df.loc[model].std()) + ")"
            results_df.loc["F1"][model] = "{:.2f}".format(f1_df.loc[model].mean()) + " (" + "{:.2f}".format(f1_df.loc[model].std()) + ")"
            

        results_df.to_csv(result_path + "/prediction_evaluation/results.tsv", sep="\t")

        generate_boxplot(to_train, num_class, auc_df, mcc_df, precision_df, recall_df, f1_df, result_path)
        generate_score_distributions(rf_scores, svm_scores, logistic_regression_scores, mlpnn_scores, label_set, result_path, to_train)           
        ranking_df = pd.DataFrame(index=range(len(features)))

        rf_ranking_df = pd.DataFrame(index=range(len(features)))
        svm_ranking_df = pd.DataFrame(index=range(len(features)))
        logistic_regression_ranking_df = pd.DataFrame(index=range(len(features)))
        mlpnn_ranking_df = pd.DataFrame(index=range(len(features)))

        model_rankings = {}
        
        if "RF" in to_train:
            for col in rf_scores.columns:
                rank_list = rf_scores[col].rank(ascending=False).sort_values(ascending=True).index.values
                ranking_df["RF_" + col] = rank_list
                rf_ranking_df["RF_" + col] = rank_list

            model_rankings["RF"] = rf_scores.median(axis=1).sort_values(ascending=False).index.values
            
        if "SVM" in to_train and num_class == 2:
            for col in svm_scores.columns:
                rank_list = svm_scores[col].abs().rank(ascending=False).sort_values(ascending=True).index.values
                ranking_df["SVM_" + col] = rank_list
                svm_ranking_df["SVM_" + col] = rank_list
        
            model_rankings["SVM"] = svm_scores.median(axis=1).sort_values(ascending=False).index.values

        elif "SVM" in to_train and num_class > 2:
            svm_joint_scores = pd.DataFrame(index=features, columns=svm_ranking_df.columns)
            for col in svm_scores[label_set[0]].columns:
                for l in range(len(label_set)):
                    if l == 0:
                        score_list = svm_scores[label_set[l]][[col]]
                    else:
                        score_list = score_list.join(svm_scores[label_set[l]][[col]], rsuffix=l)
                ranking_df["SVM_" + col] = score_list.abs().mean(axis=1).rank(ascending=False).sort_values(ascending=True).index.values
                svm_ranking_df["SVM_" + col] = score_list.abs().mean(axis=1).rank(ascending=False).sort_values(ascending=True).index.values
                svm_joint_scores["SVM_" + col] = score_list.abs().mean(axis=1)
            
            for l in range(len(label_set)):
                if l == 0:
                    score_list = svm_scores[label_set[l]].abs()
                else:
                    score_list = score_list.join(svm_scores[label_set[l]].abs(), rsuffix=l)            
            model_rankings["SVM"] = score_list.abs().mean(axis=1).sort_values(ascending=False).index.values

                
                
        
        if "Logistic Regression" in to_train and num_class == 2:
            for col in logistic_regression_scores.columns:
                rank_list = logistic_regression_scores[col].abs().rank(ascending=False).sort_values(ascending=True).index.values
                ranking_df["Logistic_Regression_" + col] = rank_list
                logistic_regression_ranking_df["Logistic_Regression_" + col] = rank_list

            model_rankings["Logistic Regression"] = logistic_regression_scores.abs().mean(axis=1).sort_values(ascending=False).index.values
            
        elif "Logistic Regression" in to_train and num_class > 2:
            logistic_regression_joint_scores = pd.DataFrame(index=features, columns=logistic_regression_ranking_df.columns)
            for col in logistic_regression_scores[label_set[0]].columns:
                for l in range(len(label_set)):
                    if l == 0:
                        score_list = logistic_regression_scores[label_set[l]][[col]]
                    else:
                        score_list = score_list.join(logistic_regression_scores[label_set[l]][[col]], rsuffix=l)
                ranking_df["Logistic Regression_" + col] = score_list.abs().mean(axis=1).rank(ascending=False).sort_values(ascending=True).index.values
                logistic_regression_ranking_df["Logistic Regression_" + col] = score_list.abs().mean(axis=1).rank(ascending=False).sort_values(ascending=True).index.values
                logistic_regression_joint_scores["Logistic Regression_" + col] = score_list.abs().mean(axis=1)
                
            for l in range(len(label_set)):
                if l == 0:
                    score_list = logistic_regression_scores[label_set[l]].abs()
                else:
                    score_list = score_list.join(logistic_regression_scores[label_set[l]].abs(), rsuffix=l)            
            model_rankings["Logistic Regression"] = score_list.mean(axis=1).sort_values(ascending=False).index.values
 
                
        if "MLPNN" in to_train:
            mlpnn_joint_scores = pd.DataFrame(index=features, columns=mlpnn_ranking_df.columns)
            for col in mlpnn_scores[label_set[0]].columns:
                for l in range(len(label_set)):
                    if l == 0:
                        score_list = mlpnn_scores[label_set[l]][[col]]
                    else:
                        score_list = score_list.join(mlpnn_scores[label_set[l]][[col]], rsuffix=l)
                ranking_df["MLPNN_" + col] = score_list.abs().mean(axis=1).rank(ascending=False).sort_values(ascending=True).index.values
                mlpnn_ranking_df["MLPNN_" + col] = score_list.abs().mean(axis=1).rank(ascending=False).sort_values(ascending=True).index.values
                mlpnn_joint_scores["MLPNN_" + col] = score_list.abs().mean(axis=1)
                
            for l in range(len(label_set)):
                if l == 0:
                    score_list = mlpnn_scores[label_set[l]].abs()
                else:
                    score_list = score_list.join(mlpnn_scores[label_set[l]].abs(), rsuffix=l)            
            model_rankings["MLPNN"] = score_list.mean(axis=1).sort_values(ascending=False).index.values
            
        if "RF" in to_train:
            rf_ranking_df.to_csv(result_path + "/feature_evaluation/rf_feature_ranks.csv")
            rf_scores.to_csv(result_path + "/feature_evaluation/rf_feature_scores.csv")
            np.savetxt(result_path + "/feature_evaluation/rf_rank_list.txt", model_rankings["RF"], fmt="%s")

        if "SVM" in to_train:
            svm_ranking_df.to_csv(result_path + "/feature_evaluation/svm_feature_ranks.csv")
            if num_class==2:
                svm_scores.to_csv(result_path + "/feature_evaluation/svm_feature_scores.csv")
            else:
                for l in label_set:
                    svm_scores[l].to_csv(result_path + "/feature_evaluation/svm_feature_scores_" + str(l) + ".csv")
            np.savetxt(result_path + "/feature_evaluation/svm_rank_list.txt", model_rankings["SVM"], fmt="%s")
                    
        if "Logistic Regression" in to_train:
            logistic_regression_ranking_df.to_csv(result_path + "/feature_evaluation/logistic_regression_feature_ranks.csv")
            if num_class==2:
                logistic_regression_scores.to_csv(result_path + "/feature_evaluation/logistic_regression_feature_scores.csv")
            else:
                for l in label_set:
                    logistic_regression_scores[l].to_csv(result_path + "/feature_evaluation/logistic_regression_feature_scores_" + str(l) + ".csv")
            np.savetxt(result_path + "/feature_evaluation/logistic_regression_rank_list.txt", model_rankings["Logistic Regression"], fmt="%s")


        if "MLPNN" in to_train:
            mlpnn_ranking_df.to_csv(result_path + "/feature_evaluation/mlpnn_feature_ranks.csv")
            for l in label_set:
                mlpnn_scores[l].to_csv(result_path + "/feature_evaluation/mlpnn_feature_scores_" + str(l) + ".csv")
            np.savetxt(result_path + "/feature_evaluation/mlpnn_rank_list.txt", model_rankings["MLPNN"], fmt="%s")


        ranking_df.to_csv(result_path + "/feature_evaluation/ensemble_rank_table.csv")
        cmd_raw = ['Rscript', 'R/aggregate_rankings.R'] + [result_path, max_k, "raw", "ensemble", agg_method]
        run_raw = subprocess.check_output(cmd_raw, universal_newlines=True)
        rank_list = np.array(pd.read_csv(result_path + "/feature_evaluation/aggregated_rank_table.csv", 
                                             index_col=None, header=None).values).reshape(-1)


       

        rank_dict = {}
        
        print("Generating training evaluation per features to use in final models...")        
        log_abundance = np.log(abundance + 1)
        
        rf_feat_auc_list = {}
        svm_feat_auc_list = {}
        logistic_regression_feat_auc_list = {}
        mlpnn_feat_auc_list = {}
                
        for num_feat in range(5,int(max_k)+1, 5):
            rf_feat_auc_list[num_feat] = []
            svm_feat_auc_list[num_feat] = []
            logistic_regression_feat_auc_list[num_feat] = []
            mlpnn_feat_auc_list[num_feat] = []

            log_abundance_sub = log_abundance[:,[x in rank_list[:num_feat] for x in features]]
            
            train_x = log_abundance_sub
            train_y = labels
            train_y_oh = labels_oh
                        
            if normalization == "MinMax":
                scaler = MinMaxScaler().fit(train_x)
                train_x = np.clip(scaler.transform(train_x), 0, 1)

            if normalization == "Standard":
                scaler = StandardScaler().fit(train_x)
                train_x = np.clip(scaler.transform(train_x), -3, 3)

            train = (train_x, train_y) 
            train_oh = (train_x, train_y_oh)
            
            rf_clf, rf_stats, _ = rf.train(train, train, config)
            svm_clf, svm_stats, _ = svm.train(train, train, config)
            lr_clf, lr_stats, _ = logistic_regression.train(train, train, config, regularization=False)
            
            mlpnn_model = MLPNN(num_feat, num_class, config, param_dict["MLPNN"])
            mlpnn_model.train(train_oh)            
            mlpnn_stats = mlpnn_model.test(train_oh)

            rf_feat_auc_list[num_feat].append(rf_stats["AUC"])
            svm_feat_auc_list[num_feat].append(svm_stats["AUC"])
            logistic_regression_feat_auc_list[num_feat].append(lr_stats["AUC"])
            mlpnn_feat_auc_list[num_feat].append(mlpnn_stats["AUC"])

        generate_feature_selection_curve(rf_feat_auc_list, svm_feat_auc_list, logistic_regression_feat_auc_list, mlpnn_feat_auc_list, result_path)
        

        
                     
        print("Generating HTML...")      
        generate_html_performance(dataset, result_path, config, to_train, results_df, label_set)  
        full_path = abspath(result_path + "/training_performance.html")
        print("Finished!")
        webbrowser.open('file://' + os.path.realpath(full_path))
        
