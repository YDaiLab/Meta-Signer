import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import interp
from sklearn.metrics import roc_curve, auc
from matplotlib_venn import venn3, venn3_circles, venn2, venn2_circles

plt.switch_backend('agg')

def generate_boxplot(to_train, num_class, auc_df, mcc_df, precision_df, recall_df, f1_df, results_path):

	color_list = []
	model_list = []
	
	first_model = True
	
	stats = {}
	stats["MCC"] = []
	stats["Precision"] = []
	stats["Recall"] = []
	stats["F1"] = []

	if num_class == 2:
		stats["AUC"] = []	

	if "RF" in to_train:
		color_list.append("Purple")
		color_list.append("Red")
		model_list.append("RF")
		model_list.append ("RF (Tree)")
		
		if num_class == 2:
			stats["AUC"].append(auc_df.loc["RF"])
		stats["MCC"].append(mcc_df.loc["RF"])
		stats["Precision"].append(precision_df.loc["RF"])
		stats["Recall"].append(recall_df.loc["RF"])
		stats["F1"].append(f1_df.loc["RF"])

		if num_class == 2:
			stats["AUC"].append(auc_df.loc["RF_TREE"])
		stats["MCC"].append(mcc_df.loc["RF_TREE"])
		stats["Precision"].append(precision_df.loc["RF_TREE"])
		stats["Recall"].append(recall_df.loc["RF_TREE"])
		stats["F1"].append(f1_df.loc["RF_TREE"])
			
	if "SVM" in to_train:
		color_list.append("Gold")
		color_list.append("Orange")
		model_list.append("SVM")
		model_list.append ("SVM (Tree)")
		
		if num_class == 2:
			stats["AUC"].append(auc_df.loc["SVM"])
		stats["MCC"].append(mcc_df.loc["SVM"])
		stats["Precision"].append(precision_df.loc["SVM"])
		stats["Recall"].append(recall_df.loc["SVM"])
		stats["F1"].append(f1_df.loc["SVM"])

		if num_class == 2:
			stats["AUC"].append(auc_df.loc["SVM_TREE"])
		stats["MCC"].append(mcc_df.loc["SVM_TREE"])
		stats["Precision"].append(precision_df.loc["SVM_TREE"])
		stats["Recall"].append(recall_df.loc["SVM_TREE"])
		stats["F1"].append(f1_df.loc["SVM_TREE"])
							
	if "LASSO" in to_train:
		color_list.append("Blue")
		color_list.append("Teal")
		model_list.append("LASSO")
		model_list.append ("LASSO (Tree)")
		
		if num_class == 2:
			stats["AUC"].append(auc_df.loc["LASSO"])
		stats["MCC"].append(mcc_df.loc["LASSO"])
		stats["Precision"].append(precision_df.loc["LASSO"])
		stats["Recall"].append(recall_df.loc["LASSO"])
		stats["F1"].append(f1_df.loc["LASSO"])

		if num_class == 2:
			stats["AUC"].append(auc_df.loc["LASSO_TREE"])
		stats["MCC"].append(mcc_df.loc["LASSO_TREE"])
		stats["Precision"].append(precision_df.loc["LASSO_TREE"])
		stats["Recall"].append(recall_df.loc["LASSO_TREE"])
		stats["F1"].append(f1_df.loc["LASSO_TREE"])
								
	if "MLPNN" in to_train:
		color_list.append("Green")
		color_list.append("Gray")
		model_list.append("MLPNN")
		model_list.append("MLPNN (TREE)")
		
		if num_class == 2:
			stats["AUC"].append(auc_df.loc["MLPNN"])
		stats["MCC"].append(mcc_df.loc["MLPNN"])
		stats["Precision"].append(precision_df.loc["MLPNN"])
		stats["Recall"].append(recall_df.loc["MLPNN"])
		stats["F1"].append(f1_df.loc["MLPNN"])

		if num_class == 2:
			stats["AUC"].append(auc_df.loc["MLPNN_TREE"])
		stats["MCC"].append(mcc_df.loc["MLPNN_TREE"])
		stats["Precision"].append(precision_df.loc["MLPNN_TREE"])
		stats["Recall"].append(recall_df.loc["MLPNN_TREE"])
		stats["F1"].append(f1_df.loc["MLPNN_TREE"])
						
	if "CNN" in to_train:
		color_list.append("Brown")
		model_list.append("CNN")
		
		if num_class == 2:
			stats["AUC"].append(auc_df.loc["CNN"])
		stats["MCC"].append(mcc_df.loc["CNN"])
		stats["Precision"].append(precision_df.loc["CNN"])
		stats["Recall"].append(recall_df.loc["CNN"])
		stats["F1"].append(f1_df.loc["CNN"])

	
	if num_class == 2:
		metric_list = ["AUC", "MCC", "Precision", "Recall", "F1"]
	else:
		metric_list = ["MCC", "Precision", "Recall", "F1"]
				

	for met in metric_list:
		fig = plt.figure(dpi=300, figsize=(16,9), tight_layout=True)
		ax = plt.subplot(111)
		ax.set_title("Boxplots of Cross-validated " + met + " Values")
		ax.set_ylabel(met)
		ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
		colors = color_list
		bp = plt.boxplot(stats[met], positions=range(1,len(model_list)+1), notch=True, widths=0.6, patch_artist=True)
		

		y_min = np.round(min(np.asarray(stats[met]).reshape(-1)) - 0.1, 1)
		y_max = np.round(max(np.asarray(stats[met]).reshape(-1)) + 0.1, 1)
		plt.xlim(0,len(model_list)+3)
		plt.ylim(y_min, y_max)
		delta = (y_max - y_min)/20
		
		for patch, fliers, col in zip(bp['boxes'], bp['fliers'], colors):
			patch.set_facecolor(col)
			plt.setp(fliers, color=col, marker='+')
		
		patch_list = []

		for i in range(0, len(model_list)):
			ax.text(i+1, y_min + delta, round(np.mean(stats[met][i]),3), horizontalalignment='center', size='large', color=colors[i])
			patch_list.append(mpatches.Patch(color=colors[i], label=model_list[i]))
			
		plt.legend(handles=patch_list, bbox_to_anchor=(1, 1), loc='upper right')
		plt.savefig(str(results_path) + "/prediction_evaluation/" + str(met) + "_boxplots.png", tight_layout=True)
		plt.clf()

def generate_auc_plot(metric_df, tpr_dict, fpr_dict, thresh_dict, total_runs, to_train, results_path):
	color_list = []
	mean_list = []
	std_list = []
	model_list = []
	
	if "RF" in to_train:
		color_list.append("Purple")
		color_list.append("Red")
		model_list.append("RF")
		model_list.append("RF (Tree)")
		mean_list.append(str(np.round(metric_df.loc["RF"].mean(), 3)))
		std_list.append(str(np.round(metric_df.loc["RF"].std(), 3)))
		mean_list.append(str(np.round(metric_df.loc["RF_TREE"].mean(), 3)))
		std_list.append(str(np.round(metric_df.loc["RF_TREE"].std(), 3)))
		
	if "SVM" in to_train:
		color_list.append("Gold")
		color_list.append("Orange")
		model_list.append("SVM")
		model_list.append("SVM (Tree)")
		mean_list.append(str(np.round(metric_df.loc["SVM"].mean(), 3)))
		std_list.append(str(np.round(metric_df.loc["SVM"].std(), 3)))
		mean_list.append(str(np.round(metric_df.loc["SVM_TREE"].mean(), 3)))
		std_list.append(str(np.round(metric_df.loc["SVM_TREE"].std(), 3)))
		
	if "LASSO" in to_train:
		color_list.append("Blue")
		color_list.append("Teal")
		model_list.append("LASSO")
		model_list.append("LASSO (Tree)")
		mean_list.append(str(np.round(metric_df.loc["LASSO"].mean(), 3)))
		std_list.append(str(np.round(metric_df.loc["LASSO"].std(), 3)))
		mean_list.append(str(np.round(metric_df.loc["LASSO_TREE"].mean(), 3)))
		std_list.append(str(np.round(metric_df.loc["LASSO_TREE"].std(), 3)))
		
	if "MLPNN" in to_train:
		color_list.append("Green")
		color_list.append("Gray")		
		model_list.append("MLPNN")
		model_list.append("MLPNN (Tree)")
		mean_list.append(str(np.round(metric_df.loc["MLPNN"].mean(), 3)))
		std_list.append(str(np.round(metric_df.loc["MLPNN"].std(), 3)))
		mean_list.append(str(np.round(metric_df.loc["MLPNN_TREE"].mean(), 3)))
		std_list.append(str(np.round(metric_df.loc["MLPNN_TREE"].std(), 3)))
		
	if "CNN" in to_train:
		color_list.append("Brown")
		model_list.append("CNN")
		mean_list.append(str(np.round(metric_df.loc["CNN"].mean(), 3)))
		std_list.append(str(np.round(metric_df.loc["CNN"].std(), 3)))
		
	fig = plt.figure(dpi=300, figsize=(9,9), tight_layout=True)
	mean_fpr = np.linspace(0, 1, total_runs)
	tprs = []
	
	for model in range(len(model_list)):
		for i in range(0,total_runs):
			tprs.append(interp(mean_fpr, fpr_dict[to_train[model]][i], tpr_dict[to_train[model]][i]))
			tprs [-1][0] = 0.0

		mean_tpr = np.mean(tprs, axis=0)
		mean_tpr[-1] = 1.0
		mean_auc = mean_list[model]
		std_auc = std_list[model]
			
		plt.plot(mean_fpr, mean_tpr, color=color_list[model],
				 label=r'Mean %s ROC (AUC = %0.2f $\pm$ %0.2f)' % (model_list[model], float(mean_auc), float(std_auc)),
				 lw=2, alpha=.8)
		std_tpr = np.std(tprs, axis=0)/10.0
		tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
		tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
		plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color_list[model], alpha=0.2)

	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Mean ROC Curves')
	plt.legend(loc="lower right")

	plt.savefig(str(results_path) + "/prediction_evaluation/ROC.png")
	return
	
def generate_score_distributions(rf_scores, rf_tree_scores, svm_scores, svm_tree_scores, lasso_scores, lasso_tree_scores, mlpnn_scores, mlpnn_tree_scores, cnn_scores, label_set, results_path, to_train):
	num_class = len(label_set)

	if "RF" in to_train:
		rf_median_scores = rf_scores.median(axis=1)

		num_bins = 50
		
		fig = plt.figure(dpi=300, figsize=(9,9), tight_layout=True)
		
		plt.ylabel("Density")
		plt.xlabel("Score")
		plt.title("RF Feature Scores")
		plt.hist(list(rf_median_scores), bins=num_bins, weights=np.ones(len(rf_median_scores)) / len(rf_median_scores), density=False)
		plt.savefig(str(results_path) + "/feature_evaluation/RF_scores.png")
		plt.clf()
		
		rf_tree_median_scores = rf_tree_scores.median(axis=1)
		
		fig = plt.figure(dpi=300, figsize=(9,9), tight_layout=True)
		plt.ylabel("Density")
		plt.xlabel("Score")
		plt.title("RF (Tree) Feature Scores")
		plt.hist(list(rf_tree_median_scores), bins=num_bins, weights=np.ones(len(rf_tree_median_scores)) / len(rf_tree_median_scores), density=False)
		plt.savefig(str(results_path) + "/feature_evaluation/RF_tree_scores.png")
		plt.clf()
		
	if "SVM" in to_train and num_class==2:
		svm_median_scores = svm_scores.median(axis=1)

		num_bins = 50
		fig = plt.figure(dpi=300, figsize=(9,9), tight_layout=True)

		plt.ylabel("Density")
		plt.xlabel("Score")
		plt.title("SVM Feature Scores")
		plt.hist(list(svm_median_scores), bins=num_bins, weights=np.ones(len(svm_median_scores)) / len(svm_median_scores), density=False)
		plt.savefig(str(results_path) + "/feature_evaluation/SVM_scores.png")
		plt.clf()

		svm_tree_median_scores = svm_tree_scores.median(axis=1)
		
		fig = plt.figure(dpi=300, figsize=(9,9), tight_layout=True)
		plt.ylabel("Density")
		plt.xlabel("Score")
		plt.title("SVM (Tree) Feature Scores")
		plt.hist(list(svm_tree_median_scores), bins=num_bins, weights=np.ones(len(svm_tree_median_scores)) / len(svm_tree_median_scores), density=False)
		plt.savefig(str(results_path) + "/feature_evaluation/SVM_tree_scores.png")
		plt.clf()


	if "LASSO" in to_train and num_class==2:
		lasso_median_scores = lasso_scores.median(axis=1)

		num_bins = 50
		
		fig = plt.figure(dpi=300, figsize=(9,9), tight_layout=True)
		
		plt.ylabel("Density")
		plt.xlabel("Score")
		plt.title("LASSO Feature Scores")
		plt.hist(list(lasso_median_scores), bins=num_bins, weights=np.ones(len(lasso_median_scores)) / len(lasso_median_scores), density=False)
		plt.savefig(str(results_path) + "/feature_evaluation/LASSO_scores.png")
		plt.clf()
		
		lasso_tree_median_scores = lasso_tree_scores.median(axis=1)
		
		fig = plt.figure(dpi=300, figsize=(9,9), tight_layout=True)
		plt.ylabel("Density")
		plt.xlabel("Score")
		plt.title("LASSO (Tree) Feature Scores")
		plt.hist(list(lasso_tree_median_scores), bins=num_bins, weights=np.ones(len(lasso_tree_median_scores)) / len(lasso_tree_median_scores), density=False)
		plt.savefig(str(results_path) + "/feature_evaluation/LASSO_tree_scores.png")
		plt.clf()	

	if "MLPNN" in to_train:
		for l in label_set:
			mlpnn_median_scores = mlpnn_scores[l].median(axis=1)
			num_bins = 50
			
			fig = plt.figure(dpi=300, figsize=(9,9), tight_layout=True)
			
			plt.ylabel("Density")
			plt.xlabel("Score")
			plt.title("MLPNN Feature Scores for " + str(l).title())
			plt.hist(list(mlpnn_median_scores), bins=num_bins, weights=np.ones(len(mlpnn_median_scores)) / len(mlpnn_median_scores), density=False)
			plt.savefig(str(results_path) + "/feature_evaluation/MLPNN_" + str(l) + "_scores.png")
			plt.clf()
			
			mlpnn_tree_median_scores =  mlpnn_tree_scores[l].median(axis=1)
			fig = plt.figure(dpi=300, figsize=(9,9), tight_layout=True)
			plt.ylabel("Density")
			plt.xlabel("Score")
			plt.title("MLPNN (Tree) Feature Scores for " + str(l).title())
			plt.hist(list(mlpnn_tree_median_scores), bins=num_bins, weights=np.ones(len(mlpnn_tree_median_scores)) / len(mlpnn_tree_median_scores), density=False)
			plt.savefig(str(results_path) + "/feature_evaluation/MLPNN_tree_" + str(l) + "_scores.png")
			plt.clf()	
			
	if "CNN" in to_train:
		for l in label_set:
			medians = cnn_scores[l].median(axis=1)
			fig = plt.figure(dpi=300, figsize=(9,9), tight_layout=True)
			plt.ylabel("Density")
			plt.xlabel("Score")
			plt.title("CNN Feature Scores for " + str(l).title())
			plt.hist(list(medians), bins=num_bins, weights=np.ones(len(medians)) / len(medians), density=False)
			plt.savefig(str(results_path) + "/feature_evaluation/CNN_" + str(l) + "_scores.png")
			plt.clf()
	return
	
def generate_venn(sets, labels, results_path):
	
	fig = plt.figure(dpi=300, figsize=(9,9), tight_layout=True)
	
	if len(sets) == 2:
		out = venn2(sets)
		for text in out.set_labels:
			text.set_fontsize(32)
		for text in out.subset_labels:
			try:
				text.set_fontsize(32)
			except:
				pass
	else:
		out = venn3(sets, labels)
		for text in out.set_labels:
			text.set_fontsize(32)
		for text in out.subset_labels:
			try:
				text.set_fontsize(32)
			except:
				pass
		
	plt.title("Feature Overlap of " +  ", ".join(labels))
	title = "venn_" + "_".join(labels) 
	plt.savefig("/home/dreiman/" + title + ".png")
	return
