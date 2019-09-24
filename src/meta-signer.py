import sys
import os
from os.path import abspath
import numpy as np
import pandas as pd
from utils.prepare_data import prepare_data
from utils.popphy_io import get_config, save_params, load_params
from utils.generate_html import generate_html
from utils.generate_plots import generate_boxplot, generate_auc_plot, generate_score_distributions, generate_venn
from utils.popphy_io import get_stat, get_stat_dict
from utils.wilcoxon_test import get_wilcoxon_ranked_list


config = get_config()

import models.rf as rf
import models.svm as svm
import models.lasso as lasso

if config.get('PopPhy', 'Train') == "True":
	from models.PopPhy import PopPhyCNN
	from utils.tune import tune_PopPhy

if config.get('MLPNN', 'Train') == "True":
	from models.mlpnn import MLPNN
	from utils.tune import tune_mlpnn

from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
from datetime import datetime
import webbrowser
import subprocess

warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.nan)

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
	train_lasso = config.get('LASSO', 'Train')
	train_mlpnn = config.get('MLPNN', 'Train')
	train_popphy = config.get('PopPhy', 'Train')
	filt_thresh = config.get('Evaluation', 'FilterThresh')
	normalization = config.get('Evaluation', 'Normalization')
	top_k = config.get('Evaluation', 'TopK')

	for dataset in datasets:

		param_tune = True
		#########################################################################
		# Read in data and generate tree maps
		#########################################################################
		print("\nStarting Meta-Signer on %s..." % (dataset))
		path = "../data/" + dataset

		my_maps, my_benchmark, my_benchmark_tree, features, tree_features, labels, label_set, g, feature_df = prepare_data(path, config)

		#labels = np.array(pd.factorize(labels)[0])
		num_class = len(np.unique(labels))
		if num_class == 2:
			metric = "AUC"
		else:
			metric = "MCC"

		seed = np.random.randint(100)
		np.random.seed(seed)
		np.random.shuffle(my_maps)
		np.random.seed(seed)
		np.random.shuffle(my_benchmark)
		np.random.seed(seed)
		np.random.shuffle(my_benchmark_tree)
		np.random.seed(seed)
		np.random.shuffle(labels)

		n_values = np.max(labels) + 1
		labels_oh = np.eye(n_values)[labels]

		print("There are %d classes...%s" % (num_class, ", ".join(label_set)))
		#########################################################################
		# Determine which models are being trained
		#########################################################################

		if num_class > 2:
			metric = "MCC"

		to_train = []

		if train_rf == "True":
			to_train.append("RF")
			to_train.append("RF_TREE")

		if train_svm == "True":
			to_train.append("SVM")
			to_train.append("SVM_TREE")

		if train_lasso == "True":
			to_train.append("LASSO")
			to_train.append("LASSO_TREE")

		if train_mlpnn == "True":
			to_train.append("MLPNN")
			to_train.append("MLPNN_TREE")

		if train_popphy == "True":
			to_train.append("PopPhy")
		#########################################################################
		# Set up DataFrames to store results
		#########################################################################

		cv_list = ["Run_" + str(x) + "_CV_" + str(y) for x in range(num_runs) for y in range(num_test)]

		auc_df = pd.DataFrame(index=to_train, columns=cv_list)
		mcc_df = pd.DataFrame(index=to_train, columns=cv_list)
		precision_df = pd.DataFrame(index=to_train, columns=cv_list)
		recall_df = pd.DataFrame(index=to_train, columns=cv_list)
		f1_df = pd.DataFrame(index=to_train, columns=cv_list)

		tpr_dict = {}
		fpr_dict = {}
		thresh_dict = {}

		for model in to_train:
			tpr_dict[model] = []
			fpr_dict[model] = []
			thresh_dict[model] = []

		#########################################################################
		# Create results directory for current run
		#########################################################################
		time_stamp = int(np.round(datetime.timestamp(datetime.now()), 0))
		ts = str(filt_thresh) + "_" + str(time_stamp)

		result_path = "../results/" + dataset + "/" + ts
		print("Saving results to %s" % (result_path))
		try:
			os.mkdir("../results/")
		except OSError:
			pass

		try:
			os.mkdir("../results/" + dataset)
		except OSError:
			pass


		try:
			os.mkdir(result_path)
		except OSError:
			print("Creation of the result subdirectory failed...")
			print("Exiting...")
			sys.exit()

		try:
			os.mkdir(result_path + "/prediction_evaluation")
		except OSError:
			print("Creation of the prediction evaluation subdirectory failed...")
			print("Exiting...")
			sys.exit()

		try:
			os.mkdir(result_path + "/feature_evaluation")
		except OSError:
			print("Creation of the feature evaluation subdirectory failed...")
			print("Exiting...")
			sys.exit()

		raw_wilcox = get_wilcoxon_ranked_list(my_benchmark, labels, features, label_set)
		tree_wilcox = get_wilcoxon_ranked_list(my_benchmark_tree, labels, tree_features, label_set)

		raw_wilcox.to_csv("../data/" + dataset + "/raw_wilcoxon_rankings.csv", sep="\t")
		tree_wilcox.to_csv("../data/" + dataset + "/tree_wilcoxon_rankings.csv", sep="\t")


		#########################################################################
		# Set up seeds for different runs
		#########################################################################

		rf_scores = pd.DataFrame(index=features)
		rf_tree_scores = pd.DataFrame(index=tree_features)
		svm_scores = pd.DataFrame(index=features)
		svm_tree_scores = pd.DataFrame(index=tree_features)
		lasso_scores = pd.DataFrame(index=features)
		lasso_tree_scores = pd.DataFrame(index=tree_features)
		mlpnn_scores = {}
		mlpnn_tree_scores = {}
		cnn_scores = {}

		for l in label_set:
			cnn_scores[l] = pd.DataFrame(index=tree_features)
			mlpnn_scores[l] = pd.DataFrame(index=features)
			mlpnn_tree_scores[l] = pd.DataFrame(index=tree_features)

		rf_rankings = pd.DataFrame(index=features)
		rf_tree_rankings = pd.DataFrame(index=tree_features)
		lasso_rankings = pd.DataFrame(index=features)
		lasso_tree_rankings = pd.DataFrame(index=tree_features)
		mlpnn_rankings = {}
		mlpnn_tree_rankings = {}
		cnn_rankings = {}

		for l in label_set:
			cnn_rankings[l] = pd.DataFrame(index=tree_features)
			mlpnn_rankings[l] = pd.DataFrame(index=features)
			mlpnn_tree_rankings[l] = pd.DataFrame(index=tree_features)

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
			for train_index, test_index in skf.split(my_benchmark, labels):

				#################################################################
				# Select and format training and testing sets
				#################################################################
				train_x, test_x = my_benchmark[train_index,:], my_benchmark[test_index,:]
				train_y, test_y = labels[train_index], labels[test_index]
				train_y_oh, test_y_oh = labels_oh[train_index,:], labels_oh[test_index,:]
				train_popphy_x, test_popphy_x = my_maps[train_index,:,:], my_maps[test_index,:,:]
				train_tree_vec, test_tree_vec = my_benchmark_tree[train_index,:], my_benchmark_tree[test_index,:]

				train_x = np.log(train_x + 1)
				test_x = np.log(test_x + 1)
				train_tree_vec = np.log(train_tree_vec + 1)
				test_tree_vec = np.log(test_tree_vec + 1)
				train_popphy_x = np.log(train_popphy_x + 1)
				test_popphy_x = np.log(test_popphy_x + 1)

				num_train_samples = train_x.shape[0]
				num_test_samples = test_x.shape[0]
				tree_row = train_popphy_x.shape[1]
				tree_col = train_popphy_x.shape[2]

				if normalization == "MinMax":
					scaler = MinMaxScaler().fit(train_x)
					train_x = np.clip(scaler.transform(train_x), 0, 1)
					test_x = np.clip(scaler.transform(test_x), 0, 1)

				if normalization == "Standard":
					scaler = StandardScaler().fit(train_x)
					train_x = np.clip(scaler.transform(train_x), -3, 3)
					test_x = np.clip(scaler.transform(test_x), -3, 3)

				if normalization == "MinMax":
					scaler = MinMaxScaler().fit(train_tree_vec)
					train_tree_vec = np.clip(scaler.transform(train_tree_vec), 0, 1)
					test_tree_vec = np.clip(scaler.transform(test_tree_vec), 0, 1)

				if normalization == "Standard":
					scaler = StandardScaler().fit(train_tree_vec)
					train_tree_vec = np.clip(scaler.transform(train_tree_vec), -3, 3)
					test_tree_vec = np.clip(scaler.transform(test_tree_vec), -3, 3)


				scaler = MinMaxScaler().fit(train_popphy_x.reshape(num_train_samples, -1))
				train_popphy_x = np.clip(scaler.transform(train_popphy_x.reshape(num_train_samples, -1)).reshape(num_train_samples, tree_row, tree_col), 0, 1)
				test_popphy_x = np.clip(scaler.transform(test_popphy_x.reshape(num_test_samples, -1)).reshape(num_test_samples, tree_row, tree_col), 0, 1)

				train_x = np.array(train_x)
				train_tree_vec = np.array(train_tree_vec)
				train_popphy_x = np.array(train_popphy_x)

				train_y = np.array(train_y)

				train = [train_x, train_y]
				test = [test_x, test_y]

				train_tree = [train_tree_vec, train_y]
				test_tree = [test_tree_vec, test_y]

				train_mlpnn_raw = [train_x, train_y_oh]
				test_mlpnn_raw = [test_x, test_y_oh]

				train_mlpnn_tree = [train_tree_vec, train_y_oh]
				test_mlpnn_tree = [test_tree_vec, test_y_oh]

				popphy_train = [train_popphy_x, train_y_oh]
				popphy_test = [test_popphy_x, test_y_oh]

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
						mlpnn_layers, mlpnn_nodes, mlpnn_lamb, mlpnn_drop = tune_mlpnn(train_mlpnn_raw, test_mlpnn_raw, config)
						param_dict["MLPNN"]["Num_Layers"] = mlpnn_layers
						param_dict["MLPNN"]["Num_Nodes"] = mlpnn_nodes
						param_dict["MLPNN"]["L2_Lambda"] = mlpnn_lamb
						param_dict["MLPNN"]["Dropout_Rate"] = mlpnn_drop

					if "MLPNN_TREE" not in param_dict and "MLPNN_TREE" in to_train:
						print("MLPNN (tree) parameters not found...Tuning MLPNN (tree) paramters...")
						param_dict["MLPNN_TREE"] = {}
						mlpnn_tree_layers, mlpnn_tree_nodes, mlpnn_tree_lamb, mlpnn_tree_drop = tune_mlpnn(train_mlpnn_tree, test_mlpnn_tree, config)
						param_dict["MLPNN_TREE"]["Num_Layers"] = mlpnn_tree_layers
						param_dict["MLPNN_TREE"]["Num_Nodes"] = mlpnn_tree_nodes
						param_dict["MLPNN_TREE"]["L2_Lambda"] = mlpnn_tree_lamb
						param_dict["MLPNN_TREE"]["Dropout_Rate"] = mlpnn_tree_drop

					if "PopPhy" not in param_dict and "PopPhy" in to_train:
						print("PopPhy parameters not found...Tuning PopPhy parameters...")
						param_dict["PopPhy"] = {}
						best_num_kernel, best_kernel_h, best_kernel_w, best_cnn_layers, best_fc_layers, best_fc_nodes, best_lambda, best_drop = tune_PopPhy(popphy_train, popphy_test, config)
						param_dict["PopPhy"]["Num_Kernel"] = best_num_kernel
						param_dict["PopPhy"]["Kernel_Height"] = best_kernel_h
						param_dict["PopPhy"]["Kernel_Width"] = best_kernel_w
						param_dict["PopPhy"]["Layers"] = best_cnn_layers	
						param_dict["PopPhy"]["FC_Layers"] = best_fc_layers
						param_dict["PopPhy"]["Num_Nodes"] = best_fc_nodes
						param_dict["PopPhy"]["L2_Lambda"] = best_lambda
						param_dict["PopPhy"]["Dropout_Rate"] = best_drop
					#############################################################
					# Save parameters
					#############################################################

					save_params(param_dict, path)



				print("\n# %s values for run %d fold %d:" % (metric, run, fold))
				print("# Model\t\t\t%s\t\tMean" % (str(metric)))

				#################################################################
				# Triain RF model using raw and tree features
				#################################################################

				if train_rf == "True":

					stats, tpr, fpr, thresh, scores, probs = rf.train(train, test, config, metric)

					rf_scores["Run_" + str(run) + "_CV_" + str(fold)] = scores

					if num_class == 2:
						auc_df.loc["RF"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["AUC"]
						tpr_dict["RF"].append(tpr)
						fpr_dict["RF"].append(fpr)
						thresh_dict["RF"].append(thresh)
					mcc_df.loc["RF"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["MCC"]
					precision_df.loc["RF"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["Precision"]
					recall_df.loc["RF"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["Recall"]
					f1_df.loc["RF"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["F1"]

					sys.stdout.write("")
					if metric == "AUC":
						print("# RF:\t\t\t%f\t%f" % (stats["AUC"], auc_df.loc["RF"].mean(axis=0)))
					if metric == "MCC":
						print("# RF:\t\t\t%f\t%f" % (stats["MCC"], mcc_df.loc["RF"].mean(axis=0)))


					stats, tpr, fpr, thresh, scores, probs = rf.train(train_tree, test_tree, config, metric)

					rf_tree_scores["Run_" + str(run) + "_CV_" + str(fold)] = scores

					if num_class == 2:
						auc_df.loc["RF_TREE"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["AUC"]
						tpr_dict["RF_TREE"].append(tpr)
						fpr_dict["RF_TREE"].append(fpr)
						thresh_dict["RF_TREE"].append(thresh)
					mcc_df.loc["RF_TREE"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["MCC"]
					precision_df.loc["RF_TREE"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["Precision"]
					recall_df.loc["RF_TREE"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["Recall"]
					f1_df.loc["RF_TREE"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["F1"]

					sys.stdout.write("")
					if metric == "AUC":
						print("# RF (Tree):\t\t%f\t%f" % (stats["AUC"], auc_df.loc["RF_TREE"].mean(axis=0)))

					if metric == "MCC":
						print("# RF (Tree):\t\t%f\t%f" % (stats["MCC"], mcc_df.loc["RF_TREE"].mean(axis=0)))


				#################################################################
				# Triain SVM model using raw and tree features
				#################################################################

				if train_svm == "True":
					stats, tpr, fpr, thresh, scores, probs = svm.train(train, test, config, metric)

					svm_scores["Run_" + str(run) + "_CV_" + str(fold)] = scores

					if num_class == 2:
						auc_df.loc["SVM"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["AUC"]
						tpr_dict["SVM"].append(tpr)
						fpr_dict["SVM"].append(fpr)
						thresh_dict["SVM"].append(thresh)
					mcc_df.loc["SVM"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["MCC"]
					precision_df.loc["SVM"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["Precision"]
					recall_df.loc["SVM"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["Recall"]
					f1_df.loc["SVM"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["F1"]

					sys.stdout.write("")
					if metric == "AUC":
						print("# SVM:\t\t\t%f\t%f" % (stats["AUC"], auc_df.loc["SVM"].mean(axis=0)))
					if metric == "MCC":
						print("# SVM:\t\t\t%f\t%f" % (stats["MCC"], mcc_df.loc["SVM"].mean(axis=0)))


					stats, tpr, fpr, thresh, scores, probs = svm.train(train_tree, test_tree, config, metric)

					svm_tree_scores["Run_" + str(run) + "_CV_" + str(fold)] = scores

					if num_class == 2:
						auc_df.loc["SVM_TREE"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["AUC"]
						tpr_dict["SVM_TREE"].append(tpr)
						fpr_dict["SVM_TREE"].append(fpr)
						thresh_dict["SVM_TREE"].append(thresh)
					mcc_df.loc["SVM_TREE"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["MCC"]
					precision_df.loc["SVM_TREE"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["Precision"]
					recall_df.loc["SVM_TREE"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["Recall"]
					f1_df.loc["SVM_TREE"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["F1"]

					sys.stdout.write("")
					if metric == "AUC":
						print("# SVM (Tree):\t\t%f\t%f" % (stats["AUC"], auc_df.loc["SVM_TREE"].mean(axis=0)))
					if metric == "MCC":
						print("# SVM (Tree):\t\t%f\t%f" % (stats["MCC"], mcc_df.loc["SVM_TREE"].mean(axis=0)))


				#################################################################
				# Triain LASSO model using raw and tree features
				#################################################################

				if train_lasso == "True":
					stats, tpr, fpr, thresh, scores, probs = lasso.train(train, test, config, metric)

					lasso_scores["Run_" + str(run) + "_CV_" + str(fold)] = scores

					if num_class == 2:
						auc_df.loc["LASSO"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["AUC"]
						tpr_dict["LASSO"].append(tpr)
						fpr_dict["LASSO"].append(fpr)
						thresh_dict["LASSO"].append(thresh)
					mcc_df.loc["LASSO"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["MCC"]
					precision_df.loc["LASSO"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["Precision"]
					recall_df.loc["LASSO"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["Recall"]
					f1_df.loc["LASSO"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["F1"]

					sys.stdout.write("")
					if metric == "AUC":
						print("# LASSO:\t\t%f\t%f" % (stats["AUC"], auc_df.loc["LASSO"].mean(axis=0)))
					if metric == "MCC":
						print("# LASSO:\t\t%f\t%f" % (stats["MCC"], mcc_df.loc["LASSO"].mean(axis=0)))


					stats, tpr, fpr, thresh, scores, probs = lasso.train(train_tree, test_tree, config, metric)

					lasso_tree_scores["Run_" + str(run) + "_CV_" + str(fold)] = scores

					if num_class == 2:
						auc_df.loc["LASSO_TREE"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["AUC"]
						tpr_dict["LASSO_TREE"].append(tpr)
						fpr_dict["LASSO_TREE"].append(fpr)
						thresh_dict["LASSO_TREE"].append(thresh)
					mcc_df.loc["LASSO_TREE"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["MCC"]
					precision_df.loc["LASSO_TREE"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["Precision"]
					recall_df.loc["LASSO_TREE"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["Recall"]
					f1_df.loc["LASSO_TREE"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["F1"]

					sys.stdout.write("")
					if metric == "AUC":
						print("# LASSO (Tree):\t\t%f\t%f" % (stats["AUC"], auc_df.loc["LASSO_TREE"].mean(axis=0)))
					if metric == "MCC":
						print("# LASSO (Tree):\t\t%f\t%f" % (stats["MCC"], mcc_df.loc["LASSO_TREE"].mean(axis=0)))


				#################################################################
				# Triain MLPNN model using raw and tree features
				#################################################################
				if train_mlpnn == "True":
					mlpnn_model = MLPNN(len(features), num_class, config, param_dict["MLPNN"])
					mlpnn_model.train(train_mlpnn_raw)
					stats, tpr, fpr, thresh, probs = mlpnn_model.test(test_mlpnn_raw)
					scores = mlpnn_model.get_scores()

					if num_class == 2:
						auc_df.loc["MLPNN"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["AUC"]
						tpr_dict["MLPNN"].append(tpr)
						fpr_dict["MLPNN"].append(fpr)
						thresh_dict["MLPNN"].append(thresh)
					mcc_df.loc["MLPNN"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["MCC"]
					precision_df.loc["MLPNN"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["Precision"]
					recall_df.loc["MLPNN"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["Recall"]
					f1_df.loc["MLPNN"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["F1"]

					sys.stdout.write("")
					if metric == "AUC":
						print("# MLPNN:\t\t%f\t%f" % (stats["AUC"], auc_df.loc["MLPNN"].mean(axis=0)))
					if metric == "MCC":
						print("# MLPNN:\t\t%f\t%f" % (stats["MCC"], mcc_df.loc["MLPNN"].mean(axis=0)))

					for l in range(len(label_set)):
						score_list = scores[:,l]
						lab = label_set[l]
						mlpnn_scores[lab]["Run_" + str(run) + "_CV_" + str(fold)] = score_list
						mlpnn_rankings[lab]["Run_" + str(run) + "_CV_" + str(fold)] = np.flip(np.argsort(score_list))

					mlpnn_tree_model = MLPNN(len(tree_features), num_class, config, param_dict["MLPNN_TREE"])
					mlpnn_tree_model.train(train_mlpnn_tree)
					stats, tpr, fpr, thresh, probs = mlpnn_tree_model.test(test_mlpnn_tree)
					scores = mlpnn_tree_model.get_scores()

					if num_class == 2:
						auc_df.loc["MLPNN_TREE"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["AUC"]
						tpr_dict["MLPNN_TREE"].append(tpr)
						fpr_dict["MLPNN_TREE"].append(fpr)
						thresh_dict["MLPNN_TREE"].append(thresh)
					mcc_df.loc["MLPNN_TREE"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["MCC"]
					precision_df.loc["MLPNN_TREE"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["Precision"]
					recall_df.loc["MLPNN_TREE"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["Recall"]
					f1_df.loc["MLPNN_TREE"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["F1"]

					sys.stdout.write("")
					if metric == "AUC":
						print("# MLPNN (Tree):\t\t%f\t%f" % (stats["AUC"], auc_df.loc["MLPNN_TREE"].mean(axis=0)))
					if metric == "MCC":
						print("# MLPNN (Tree):\t\t%f\t%f" % (stats["MCC"], mcc_df.loc["MLPNN_TREE"].mean(axis=0)))


					for l in range(len(label_set)):
						score_list = scores[:,l]
						lab = label_set[l]
						mlpnn_tree_scores[lab]["Run_" + str(run) + "_CV_" + str(fold)] = score_list
						mlpnn_tree_rankings[lab]["Run_" + str(run) + "_CV_" + str(fold)] = np.flip(np.argsort(score_list))


				#################################################################
				# Triain CNN model using tree maps
				#################################################################

				if train_popphy == "True":
					popphy_model = PopPhyCNN((popphy_train[0].shape[1],popphy_train[0].shape[2]), num_class, config, param_dict["PopPhy"])
					popphy_model.train(popphy_train)
					stats, tpr, fpr, thresh, probs = popphy_model.test(popphy_test)
					scores = popphy_model.get_feature_scores(popphy_train, g, label_set, tree_features, config)

					if num_class == 2:
						auc_df.loc["PopPhy"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["AUC"]
						tpr_dict["PopPhy"].append(tpr)
						fpr_dict["PopPhy"].append(fpr)
						thresh_dict["PopPhy"].append(thresh)
					mcc_df.loc["PopPhy"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["MCC"]
					precision_df.loc["PopPhy"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["Precision"]
					recall_df.loc["PopPhy"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["Recall"]
					f1_df.loc["PopPhy"]["Run_" + str(run) + "_CV_" + str(fold)]=stats["F1"]

					sys.stdout.write("")
					if metric == "AUC":
						print("# CNN:\t\t\t%f\t%f" % (stats["AUC"], auc_df.loc["PopPhy"].mean(axis=0)))
					if metric == "MCC":
						print("# CNN:\t\t\t%f\t%f" % (stats["MCC"], mcc_df.loc["PopPhy"].mean(axis=0)))

					for l in range(len(label_set)):
						score_list = scores[:,l]
						lab = label_set[l]
						cnn_scores[lab]["Run_" + str(run) + "_CV_" + str(fold)] = score_list
						cnn_rankings[lab]["Run_" + str(run) + "_CV_" + str(fold)] = np.flip(np.argsort(score_list))

				fold += 1
			run += 1
			#####################################################################
			# Save metric dataframes as files
			#####################################################################

		if num_class == 2:
			auc_df.to_csv(result_path + "/prediction_evaluation/results_AUC.tsv", sep="\t")
		mcc_df.to_csv(result_path + "/prediction_evaluation/results_MCC.tsv", sep="\t")
		precision_df.to_csv(result_path + "/prediction_evaluation/results_precision.tsv", sep="\t")
		recall_df.to_csv(result_path + "/prediction_evaluation/results_recall.tsv", sep="\t")
		f1_df.to_csv(result_path + "/prediction_evaluation/results_F1.tsv", sep="\t")

		#####################################################################
		# Store mean and std of metrics as file
		#####################################################################

		if num_class == 2:
			results_df = pd.DataFrame(index=["AUC", "MCC", "Precision", "Recall", "F1"], columns=to_train)
		else:
			results_df = pd.DataFrame(index=["MCC", "Precision", "Recall", "F1"], columns=to_train)

		for model in to_train:
			if num_class == 2:
				results_df.loc["AUC"][model] = str(np.round(auc_df.loc[model].mean(), 3)) + " (" + str(np.round(auc_df.loc[model].std(), 3)) + ")"
			results_df.loc["MCC"][model] = str(np.round(mcc_df.loc[model].mean(), 3)) + " (" + str(np.round(mcc_df.loc[model].std(), 3)) + ")"
			results_df.loc["Precision"][model] = str(np.round(precision_df.loc[model].mean(), 3)) + " (" + str(np.round(precision_df.loc[model].std(), 3)) + ")"
			results_df.loc["Recall"][model] = str(np.round(recall_df.loc[model].mean(), 3)) + " (" + str(np.round(recall_df.loc[model].std(), 3)) + ")"
			results_df.loc["F1"][model] = str(np.round(f1_df.loc[model].mean(), 3)) + " (" + str(np.round(f1_df.loc[model].std(), 3)) + ")"


		for l in label_set:
			cnn_scores[l] = cnn_scores[l].fillna(0)

		results_df.to_csv(result_path + "/prediction_evaluation/results.tsv", sep="\t")

		generate_boxplot(to_train, num_class, auc_df, mcc_df, precision_df, recall_df, f1_df, result_path)

		if num_class == 2:
			generate_auc_plot(auc_df, tpr_dict, fpr_dict, thresh_dict, num_runs*num_test, to_train, result_path)

		generate_score_distributions(rf_scores, rf_tree_scores, svm_scores, svm_tree_scores, lasso_scores, lasso_tree_scores, mlpnn_scores, mlpnn_tree_scores, cnn_scores, label_set, result_path, to_train)

		ranking_df = pd.DataFrame(index=range(len(features)))
		tree_ranking_df = pd.DataFrame(index=range(len(tree_features)))

		rf_ranking_df = pd.DataFrame(index=range(len(features)))
		rf_tree_ranking_df = pd.DataFrame(index=range(len(tree_features)))

		svm_ranking_df = pd.DataFrame(index=range(len(features)))
		svm_tree_ranking_df = pd.DataFrame(index=range(len(tree_features)))

		lasso_ranking_df = pd.DataFrame(index=range(len(features)))
		lasso_tree_ranking_df = pd.DataFrame(index=range(len(tree_features)))

		mlpnn_ranking_df = pd.DataFrame(index=range(len(features)))
		mlpnn_tree_ranking_df = pd.DataFrame(index=range(len(tree_features)))

		cnn_tree_ranking_df = pd.DataFrame(index=range(len(tree_features)))


		if "species" in feature_df.columns.values:
			token = "s__"
		else:
			token = "g__"

		tree_ranking_set = {}
		for col in feature_df.columns.values:
			tree_ranking_set[col] = {}
		tree_ranking_set["Other"] = {}

		if "RF" in to_train:
			for col in rf_scores.columns:
				rank_list = rf_scores[col].rank(ascending=False).sort_values(ascending=True).index.values
				ranking_df["RF_" + col] = rank_list
				rf_ranking_df["RF_" + col] = rank_list

			for col in rf_tree_scores.columns:
				rank_list = rf_tree_scores[col].rank(ascending=False).sort_values(ascending=True).index.values
				tree_ranking_df["RF_" + col] = rank_list
				rf_tree_ranking_df["RF_" + col] = rank_list

		if "SVM" in to_train and num_class == 2:
			for col in svm_scores.columns:
				rank_list = svm_scores[col].abs().rank(ascending=False).sort_values(ascending=True).index.values
				ranking_df["SVM_" + col] = rank_list
				svm_ranking_df["SVM_" + col] = rank_list

			for col in svm_tree_scores.columns:
				rank_list = svm_tree_scores[col].abs().rank(ascending=False).sort_values(ascending=True).index.values
				tree_ranking_df["SVM_" + col] = rank_list
				svm_tree_ranking_df["SVM_" + col] = rank_list

		if "LASSO" in to_train and num_class == 2:
			for col in lasso_scores.columns:
				rank_list = lasso_scores[col].abs().rank(ascending=False).sort_values(ascending=True).index.values
				ranking_df["LASSO_" + col] = rank_list
				lasso_ranking_df["LASSO_" + col] = rank_list

			for col in lasso_tree_scores.columns:
				rank_list = lasso_tree_scores[col].abs().rank(ascending=False).sort_values(ascending=True).index.values
				tree_ranking_df["LASSO_" + col] = rank_list
				lasso_tree_ranking_df["LASSO_" + col] = rank_list

		if "MLPNN" in to_train:
			for col in mlpnn_scores[label_set[0]].columns:
				for l in range(len(label_set)):
					if l == 0:
						score_list = mlpnn_scores[label_set[l]][[col]]
					else:
						score_list = score_list.join(mlpnn_scores[label_set[l]][[col]], rsuffix=l)
				ranking_df["MLPNN_" + col] = score_list.max(axis=1).rank(ascending=False).sort_values(ascending=True).index.values
				mlpnn_ranking_df["MLPNN_" + col] = score_list.max(axis=1).rank(ascending=False).sort_values(ascending=True).index.values

			for col in mlpnn_tree_scores[label_set[0]].columns:
				for l in range(len(label_set)):
					if l == 0:
						score_list = mlpnn_tree_scores[label_set[l]][[col]]
					else:
						score_list = score_list.join(mlpnn_tree_scores[label_set[l]][[col]], rsuffix=l)
				tree_ranking_df["MLPNN_" + col] = score_list.max(axis=1).rank(ascending=False).sort_values(ascending=True).index.values
				mlpnn_tree_ranking_df["MLPNN_" + col] = score_list.max(axis=1).rank(ascending=False).sort_values(ascending=True).index.values

		col_num = 0
		if "PopPhy" in to_train:
			for col in cnn_scores[label_set[0]].columns:
				for l in range(len(label_set)):
					if l == 0:
						score_list = cnn_scores[label_set[l]].reindex(tree_features)[[col]]
					else:
						score_list = score_list.join(cnn_scores[label_set[l]][[col]], rsuffix=l)
				col_run = str(int(col_num / num_runs))
				col_cv = str(int(col_num % num_test))
				new_col = "Run_" + str(col_run) + "_CV_" + str(col_cv)

				tree_ranking_df["CNN_" + new_col] = score_list.max(axis=1).rank(ascending=False).sort_values(ascending=True).index.values
				cnn_tree_ranking_df["CNN_" + new_col] = score_list.max(axis=1).rank(ascending=False).sort_values(ascending=True).index.values
				col_num += 1

		if "RF" in to_train:
			rf_ranking_df.to_csv(result_path + "/feature_evaluation/rf_raw_rank_table.csv")
			rf_tree_ranking_df.to_csv(result_path + "/feature_evaluation/rf_tree_rank_table.csv")

		if "SVM" in to_train and num_class == 2:
			svm_ranking_df.to_csv(result_path + "/feature_evaluation/svm_raw_rank_table.csv")
			svm_tree_ranking_df.to_csv(result_path + "/feature_evaluation/svm_tree_rank_table.csv")

		if "LASSO" in to_train and num_class == 2:
			lasso_ranking_df.to_csv(result_path + "/feature_evaluation/lasso_raw_rank_table.csv")
			lasso_tree_ranking_df.to_csv(result_path + "/feature_evaluation/lasso_tree_rank_table.csv")

		if "MLPNN" in to_train:
			mlpnn_ranking_df.to_csv(result_path + "/feature_evaluation/mlpnn_raw_rank_table.csv")
			mlpnn_tree_ranking_df.to_csv(result_path + "/feature_evaluation/mlpnn_tree_rank_table.csv")

		if "PopPhy" in to_train:
			cnn_tree_ranking_df.to_csv(result_path + "/feature_evaluation/cnn_tree_rank_table.csv")

		if ("LASSO" in to_train and num_class == 2) or "MLPNN" in to_train or "RF" in to_train:
			ranking_df.to_csv(result_path + "/feature_evaluation/ensemble_raw_rank_table.csv")

		if ("LASSO" in to_train and num_class == 2) or "MLPNN" in to_train or "RF" in to_train or "PopPhy" in to_train:
			tree_ranking_df.to_csv(result_path + "/feature_evaluation/ensemble_tree_rank_table.csv")

		if ("LASSO" in to_train and num_class == 2) or "MLPNN" in to_train or "RF" in to_train:
			cmd_raw = ['Rscript', 'R/aggregate_rankings.R'] + [result_path, top_k, "raw", "ensemble", metric]
			run_raw = subprocess.check_output(cmd_raw, universal_newlines=True)

		if ("LASSO" in to_train and num_class == 2) or "MLPNN" in to_train or "RF" in to_train or "PopPhy" in to_train:
			cmd_tree = ['Rscript', 'R/aggregate_rankings.R'] + [result_path, top_k, "tree", "ensemble", metric]
			run_tree = subprocess.check_output(cmd_tree, universal_newlines=True)

		if ("LASSO" in to_train and num_class == 2) or "MLPNN" in to_train or "RF" in to_train:
			rank_list = np.array(pd.read_csv(result_path + "/feature_evaluation/ensemble_raw_rank_table_aggregated.csv", index_col=None, header=None).values).reshape(-1)

		if ("LASSO" in to_train and num_class == 2) or "MLPNN" in to_train or "RF" in to_train or "PopPhy" in to_train:
			tree_rank_list = np.array(pd.read_csv(result_path + "/feature_evaluation/ensemble_tree_rank_table_aggregated.csv", index_col=None, header=None).values).reshape(-1)

		for feat in tree_rank_list:
			found=False
			feat = feat.replace("_species", "").replace("_genus", "").replace("_family", "").replace("_order", "").replace("_class", "").replace("_phylum", "").replace("_genus", "")
			if "unclassified" in feat:
				feat = feat.replace("unclassified_","")

			for col in feature_df.columns.values:
				if not found:
					if feat in feature_df[col].values:
						found=True
						if feat in tree_ranking_set[col]:
							tree_ranking_set[col][feat] += 1
						else:
							tree_ranking_set[col][feat] = 1
			if not found:
				col = "Other"
				if feat in tree_ranking_set[col]:
					tree_ranking_set[col][feat] += 1
				else:
					tree_ranking_set[col][feat] = 1

		rank_dict = {}
		tree_rank_dict = {}

		raw_wilcox = raw_wilcox.sort_values(raw_wilcox.columns[0])
		tree_wilcox = tree_wilcox.sort_values(by=tree_wilcox.columns[0])

		for feat in rank_list:
			rank_dict[feat] = {}
			count = np.count_nonzero(ranking_df.head(int(top_k)).values == feat)
			rank_dict[feat]["percent"] = str(np.round(float(count) / float(len(ranking_df.columns)),2))
			rank_dict[feat]["Wilcoxon_p"] = '{:.3e}'.format(raw_wilcox.loc[feat].values[0])
			rank_dict[feat]["Wilcoxon_rank"] = str(raw_wilcox.index.get_loc(feat) + 1)
			enriched_lab="-"
			cl_vote = 0
			cl_list = []

			if "SVM" in to_train and num_class == 2:
				if np.median(svm_scores.loc[feat].values) > np.quantile(svm_scores.loc[feat].values, 0.75):
					cl_vote = cl_vote + 1
				elif np.median(svm_scores.loc[feat].values) < np.quantile(svm_scores.loc[feat].values, 0.25):
					cl_vote = cl_vote - 1

			if "LASSO" in to_train and num_class == 2:
				if np.median(lasso_scores.loc[feat].values) > np.quantile(lasso_scores.loc[feat].values, 0.75):
					cl_vote = cl_vote + 1
				elif np.median(lasso_scores.loc[feat].values) < np.quantile(lasso_scores.loc[feat].values, 0.25):
					cl_vote = cl_vote - 1

			if "MLPNN" in to_train:
				if num_class == 2:
					if np.median(mlpnn_scores[label_set[0]].loc[feat].values) > np.median(mlpnn_scores[label_set[1]].loc[feat].values):
						cl_vote = cl_vote - 1
					else:
						cl_vote = cl_vote + 1
				else:
					max_val = 0
					enriched_lab = "-"
					for lab in label_set:
						if np.median(mlpnn_scores[lab].loc[feat].values) > max_val:
							enriched_lab = lab

			if num_class == 2:
				if cl_vote > 0:
					cl_list = label_set[1]
				elif cl_vote < 0:
					cl_list = label_set[0]
				else:
					cl_list = "-"
				rank_dict[feat]["Enriched"] = cl_list

			else:
				rank_dict[feat]["Enriched"] = enriched_lab

		for feat in tree_rank_list:
			feat_trim = feat.replace("_species", "").replace("_genus", "").replace("_family", "").replace("_order", "").replace("_class", "").replace("_phylum", "").replace("_genus", "")
			if "unclassified" in feat_trim:
				feat_trim = feat_trim.replace("unclassified_","")
			tree_rank_dict[feat_trim] = {}
			count = np.count_nonzero(tree_ranking_df.head(int(top_k)).values == feat)
			tree_rank_dict[feat_trim]["percent"] = str(np.round(float(count) / float(len(ranking_df.columns)),2))
			tree_rank_dict[feat_trim]["Wilcoxon_p"] = '{:.3e}'.format(tree_wilcox.loc[feat].values[0])
			tree_rank_dict[feat_trim]["Wilcoxon_rank"] = str(tree_wilcox.index.get_loc(feat) + 1)

			enriched_lab="-"
			cnn_enriched_lab = "-"
			cl_vote = 0
			cl_list = []

			if "SVM" in to_train and num_class == 2:
				if np.median(svm_tree_scores.loc[feat].values) > np.quantile(svm_tree_scores.loc[feat].values, 0.75):
					cl_vote = cl_vote + 1
				elif np.median(svm_tree_scores.loc[feat].values) < np.quantile(svm_tree_scores.loc[feat].values, 0.25):
					cl_vote = cl_vote - 1
			if "LASSO" in to_train and num_class == 2:
				if np.median(lasso_tree_scores.loc[feat].values) > np.quantile(lasso_tree_scores.loc[feat].values, 0.75):
					cl_vote = cl_vote + 1
				elif np.median(lasso_tree_scores.loc[feat].values) < np.quantile(lasso_tree_scores.loc[feat].values, 0.25):
					cl_vote = cl_vote - 1

			if "MLPNN" in to_train:
				if num_class == 2:
					if np.median(mlpnn_tree_scores[label_set[0]].loc[feat].values) > np.median(mlpnn_tree_scores[label_set[1]].loc[feat].values):
						cl_vote = cl_vote - 1
					else:
						cl_vote = cl_vote + 1
				else:
					max_val = 0
					enriched_lab = "-"
					for lab in label_set:
						if np.median(mlpnn_tree_scores[lab].loc[feat].values) > max_val:
								enriched_lab = lab
			'''
			if "PopPhy" in to_train:
				if num_class == 2:
					if np.median(cnn_scores[label_set[0]].loc[feat].values) > np.median(cnn_scores[label_set[1]].loc[feat].values):
						cl_vote = cl_vote - 1
					else:
						cl_vote = cl_vote + 1
				else:
					max_val = 0
					enriched_lab = "-"
					for lab in label_set:
						if np.median(cnn_scores[lab].loc[feat].values) > max_val:
							cnn_enriched_lab = lab
			'''
			if num_class == 2:
				if cl_vote > 0:
					cl_list = label_set[1]
				elif cl_vote < 0:
					cl_list = label_set[0]
				else:
					cl_list = "-"
				tree_rank_dict[feat_trim]["Enriched"] = cl_list
			else:
				if "MLPNN" in to_train:
					tree_rank_dict[feat_trim]["Enriched"] = enriched_lab
				elif "PopPhy" in to_train:
					tree_rank_dict[feat_trim]["Enriched"] = cnn_enriched_lab
		generate_html(dataset, result_path, config, to_train, time_stamp, results_df, label_set, rank_list, tree_ranking_set, rank_dict, tree_rank_dict)


		full_path = abspath(result_path + "/results.html")
		webbrowser.open('file://' + os.path.realpath(full_path))
