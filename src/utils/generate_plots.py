import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import interp
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import pandas as pd

plt.switch_backend('agg')

def generate_boxplot(to_train, num_class, auc_df, mcc_df, precision_df, recall_df, f1_df, results_path):

    fig = plt.figure(figsize=(8,6), dpi=300)
    sns.boxplot(data=pd.melt(auc_df.transpose()), x="variable", y="value")
    plt.xlabel('')
    plt.ylabel('AUC ROC')
    plt.title('Cross-validated AUC Performance')    
    plt.savefig(str(results_path) + "/prediction_evaluation/auc_boxplots.png", tight_layout=True)
    plt.clf()
    
    fig = plt.figure(figsize=(8,6), dpi=300)
    sns.boxplot(data=pd.melt(mcc_df.transpose()), x="variable", y="value")
    plt.xlabel('')
    plt.ylabel('MCC')
    plt.title('Cross-validated MCC Performance') 
    plt.savefig(str(results_path) + "/prediction_evaluation/mcc_boxplots.png", tight_layout=True)
    plt.clf()

    fig = plt.figure(figsize=(8,6), dpi=300)
    sns.boxplot(data=pd.melt(precision_df.transpose()), x="variable", y="value")
    plt.xlabel('')
    plt.ylabel('Precision')
    plt.title('Cross-validated Precision Performance') 
    plt.savefig(str(results_path) + "/prediction_evaluation/precision_boxplots.png", tight_layout=True)
    plt.clf()
    
    fig = plt.figure(figsize=(8,6), dpi=300)    
    sns.boxplot(data=pd.melt(recall_df.transpose()), x="variable", y="value")
    plt.xlabel('')
    plt.ylabel('Recall')
    plt.title('Cross-validated Recall Performance') 
    plt.savefig(str(results_path) + "/prediction_evaluation/recall_boxplots.png", tight_layout=True)
    plt.clf()
    
    fig = plt.figure(figsize=(8,6), dpi=300)    
    sns.boxplot(data=pd.melt(f1_df.transpose()), x="variable", y="value")
    plt.xlabel('')
    plt.ylabel('F1-Score')
    plt.title('Cross-validated F1-Score Performance') 
    plt.savefig(str(results_path) + "/prediction_evaluation/f1_boxplots.png", tight_layout=True)
    plt.clf()
    
def generate_feature_selection_curve(rf_list, svm_list, lr_list,mlpnn_list, results_path):
    plt.figure(figsize=(8,8), dpi=300)
    plot_df = pd.DataFrame(columns=["Method", "AUC", "Number of Features"])
    plot_df = plot_df.append({"Method":"RF", "AUC":0.5, "Number of Features":0}, ignore_index=True)
    plot_df = plot_df.append({"Method":"SVM", "AUC":0.5, "Number of Features":0}, ignore_index=True)
    plot_df = plot_df.append({"Method":"Logistic Regression", "AUC":0.5, "Number of Features":0}, ignore_index=True)
    plot_df = plot_df.append({"Method":"MLPNN", "AUC":0.5, "Number of Features":0}, ignore_index=True)

    for m in rf_list:
        plot_df = plot_df.append({"Method":"RF", "AUC":np.mean(rf_list[m]), "Number of Features":m}, ignore_index=True)
        plot_df = plot_df.append({"Method":"SVM", "AUC":np.mean(svm_list[m]), "Number of Features":m}, ignore_index=True)
        plot_df = plot_df.append({"Method":"Logistic Regression", "AUC":np.mean(lr_list[m]), "Number of Features":m}, ignore_index=True)
        plot_df = plot_df.append({"Method":"MLPNN", "AUC":np.mean(mlpnn_list[m]), "Number of Features":m}, ignore_index=True)


    sns.lineplot(data=plot_df, x="Number of Features", y="AUC", hue="Method")
    plt.savefig(str(results_path) + "/feature_evaluation/feature_selection.png", tight_layout=True)

    

def generate_score_distributions(rf_scores, svm_scores, lr_scores, mlpnn_scores, label_set, results_path, to_train):

    num_class = len(label_set)

    if "RF" in to_train:
        rf_median_scores = rf_scores.median(axis=1)        
        fig = plt.figure(dpi=300, figsize=(9,9), tight_layout=True)
        
        plt.ylabel("Density")
        plt.xlabel("Score")
        plt.title("RF Feature Scores")
        sns.distplot(list(rf_median_scores))
        plt.savefig(str(results_path) + "/feature_evaluation/RF_scores.png")
        plt.clf()

    if "SVM" in to_train:
        fig = plt.figure(dpi=300, figsize=(9,9), tight_layout=True)

        plt.ylabel("Density")
        plt.xlabel("Score")
        plt.title("SVM Feature Scores")
        if num_class == 2:
            svm_median_scores = svm_scores.median(axis=1).values
            sns.distplot(svm_median_scores)
        else:
            for l in label_set:
                svm_median_scores = svm_scores[l].median(axis=1).values  
                sns.distplot(svm_median_scores, label=str(l).title())
            plt.legend()
        plt.savefig(str(results_path) + "/feature_evaluation/SVM_scores.png")
        plt.clf()


    if "Logistic Regression" in to_train:
        fig = plt.figure(dpi=300, figsize=(9,9), tight_layout=True)
        
        plt.ylabel("Density")
        plt.xlabel("Score")
        plt.title("Logistic Regression Feature Scores")
        if num_class == 2:
            lr_median_scores = lr_scores.median(axis=1).values
            sns.distplot(lr_median_scores,  kde_kws={'bw':0.1})
        else:
            for l in label_set:
                lr_median_scores = lr_scores[l].median(axis=1).values
                sns.distplot(lr_median_scores, label=str(l).title(),  kde_kws={'bw':0.1})
            plt.legend()        
        plt.savefig(str(results_path) + "/feature_evaluation/Logistic_Regression_scores.png")
        plt.clf()

    if "MLPNN" in to_train:
        fig = plt.figure(dpi=300, figsize=(9,9), tight_layout=True)
        plt.ylabel("Density")
        plt.xlabel("Score")
        plt.title("MLPNN Feature Scores")
        for l in label_set:
            mlpnn_median_scores = mlpnn_scores[l].median(axis=1)            
            sns.distplot(list(mlpnn_median_scores), label=str(l).title())
        plt.legend()
        plt.savefig(str(results_path) + "/feature_evaluation/MLPNN_scores.png")
        plt.clf()

    return
    
