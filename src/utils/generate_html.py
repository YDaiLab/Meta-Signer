import numpy as np
import os
import struct
import base64
from array import array as pyarray
from numpy import unique
from utils.graph import Graph
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd
from datetime import datetime

def generate_html(dataset, path, config, to_train, time_stamp, results_df, label_set, ranking_set, tree_ranking_set, rank_dict, tree_rank_dict):
	html_file = open(path + "/results.html", "w")
	filter_thresh = config.get('Evaluation', 'FilterThresh')
	test_splits = config.get('Evaluation', 'NumberTestSplits')
	num_runs = config.get('Evaluation', 'NumberRuns')
	roc_b64_str = ""
	auc_bp_b64_str = ""
	mcc_bp_b64_str = ""
	rec_bp_b64_str = ""
	pre_bp_b64_str = ""
	f1_bp_b64_str = ""

	with open(path+"/prediction_evaluation/MCC_boxplots.png", "rb") as image_file:
		mcc_bp_b64_str = str(base64.b64encode(image_file.read()).decode('utf-8').replace('\n', ''))
	
	with open(path+"/prediction_evaluation/Recall_boxplots.png", "rb") as image_file:
		rec_bp_b64_str = str(base64.b64encode(image_file.read()).decode('utf-8').replace('\n', ''))

	with open(path+"/prediction_evaluation/Precision_boxplots.png", "rb") as image_file:
		pre_bp_b64_str = str(base64.b64encode(image_file.read()).decode('utf-8').replace('\n', ''))

	with open(path+"/prediction_evaluation/F1_boxplots.png", "rb") as image_file:
		f1_bp_b64_str = str(base64.b64encode(image_file.read()).decode('utf-8').replace('\n', ''))
	
	date = datetime.fromtimestamp(time_stamp)
	num_class = len(label_set)
	if num_class > 2:
		metric = "MCC"
	else:
		metric = "AUC"
	
	model_string = ""
	results_table_string = """<div style="margin-bottom:50px;"><table class="model_table"><tr><th style="width:150px;text-align:left">Model</th><th style="width:100px">AUC</th><th style="width:100px">MCC</th><th style="width:100px">Precision</th><th style="width:100px">Recall</th><th style="width:100px">F1</th></tr>"""
	
	for m in to_train:
		if m == "RF" or m == "SVM" or m == "LASSO" or m == "MLPNN" or m == "CNN":
			model_string = model_string + m + "&emsp;"
			if num_class == 2:
				mean_auc = np.round(float(results_df.loc["AUC"][m].split("(")[0]), 3)
			else:
				mean_auc = "-"
			mean_mcc = np.round(float(results_df.loc["MCC"][m].split("(")[0]), 3)
			mean_prec = np.round(float(results_df.loc["Precision"][m].split("(")[0]), 3)
			mean_recall = np.round(float(results_df.loc["Recall"][m].split("(")[0]), 3)
			mean_f1 = np.round(float(results_df.loc["F1"][m].split("(")[0]), 3)
			
			results_table_string += """<tr><td style="text-align:left">""" + m + "</td><td>" + str(mean_auc) + "</td><td>" + str(mean_mcc) + "</td><td>" + str(mean_prec) + "</td><td>" + str(mean_recall) + "</td><td>" + str(mean_f1) + "</td></tr>"
			
			if m != "CNN":
				m_tree = m + "_TREE"
				if num_class == 2:
					mean_auc = np.round(float(results_df.loc["AUC"][m_tree].split("(")[0]), 3)
				else:
					mean_auc = "-"
				mean_mcc = np.round(float(results_df.loc["MCC"][m_tree].split("(")[0]), 3)
				mean_prec = np.round(float(results_df.loc["Precision"][m_tree].split("(")[0]), 3)
				mean_recall = np.round(float(results_df.loc["Recall"][m_tree].split("(")[0]), 3)
				mean_f1 = np.round(float(results_df.loc["F1"][m_tree].split("(")[0]), 3)
				
				results_table_string += """<tr><td style="text-align:left">""" + m + " (TREE)</td><td>" + str(mean_auc) + "</td><td>" + str(mean_mcc) + "</td><td>" + str(mean_prec) + "</td><td>" + str(mean_recall) + "</td><td>" + str(mean_f1) + "</td></tr>"
		
	
	results_table_string += "</table>"
	if num_class == 2:
		with open(path+"/prediction_evaluation/ROC.png", "rb") as image_file:
			roc_b64_str = str(base64.b64encode(image_file.read()).decode('ascii'))

		results_table_string+="""<img src="data:image/png;base64,""" + roc_b64_str + """\" style="width:30%" class="border">"""
	results_table_string+="""</div>"""




	if num_class == 2:
		with open(path+"/prediction_evaluation/AUC_boxplots.png", "rb") as image_file:
			auc_bp_b64_str = str(base64.b64encode(image_file.read()).decode('utf-8').replace('\n', ''))

		auc_boxplot_button ="""<td><button class="button" onclick="openBoxplot('AUC')">AUC</button></td>"""
		auc_boxplot_str = """<div id="AUC" class="w3-container boxplot"><img src="data:image/png;base64,""" + auc_bp_b64_str + """\" style="width:80%" class="border"></div>"""
		mcc_display = """ style="display:none;" """
	else:
		auc_boxplot_button = ""
		auc_boxplot_str = ""
		mcc_display = ""

	feature_dist_string = "<div style=width:100%;padding-bottom:10px;><table><tr>"
	
	for m in to_train:
		if m == "RF":
			feature_dist_string += """<td><button class="button" onclick="openDist('RF')">RF</button></td>"""
			feature_dist_string += """<td><button class="button" onclick="openDist('RF_TREE')">RF (TREE)</button></td>"""
		if m == "SVM" and num_class==2:
			feature_dist_string += """<td><button class="button" onclick="openDist('SVM')">SVM</button></td>"""
			feature_dist_string += """<td><button class="button" onclick="openDist('SVM_TREE')">SVM (TREE)</button></td>"""
		if m == "LASSO" and num_class==2:
			feature_dist_string += """<td><button class="button" onclick="openDist('LASSO')">LASSO</button></td>"""
			feature_dist_string += """<td><button class="button" onclick="openDist('LASSO_TREE')">LASSO (TREE)</button></td>"""
		if m == "MLPNN":
			feature_dist_string += """<td><button class="button" onclick="openDist('MLPNN')">MLPNN</button></td>"""
			feature_dist_string += """<td><button class="button" onclick="openDist('MLPNN_TREE')">MLPNN (TREE)</button></td>"""
		if m == "CNN":
			feature_dist_string += """<td><button class="button" onclick="openDist('CNN')">CNN</button></td>"""
	feature_dist_string += "</tr></table></div>"
	
	display_str = ""
	for m in to_train:
	
		if m == "RF":
			with open(path+"/feature_evaluation/RF_scores.png", "rb") as image_file:
				b64_str = str(base64.b64encode(image_file.read()).decode('utf-8').replace('\n', ''))
			feature_dist_string += """<div id="RF" class="w3-container feature_dist" style='""" + display_str + """ '><img src="data:image/png;base64,""" + b64_str + """\" style="width:50%;""" + display_str + """" class="border"></div>"""
		
			with open(path+"/feature_evaluation/RF_tree_scores.png", "rb") as image_file:
				b64_str = str(base64.b64encode(image_file.read()).decode('utf-8').replace('\n', ''))
			feature_dist_string += """<div id="RF_TREE" class="w3-container feature_dist" style="display:none;"><img src="data:image/png;base64,""" + b64_str + """\" style="width:50%;" class="border"></div>"""
			display_str = "display:none;"
		
		if m == "SVM" and num_class==2:
			with open(path+"/feature_evaluation/SVM_scores.png", "rb") as image_file:
				b64_str = str(base64.b64encode(image_file.read()).decode('utf-8').replace('\n', ''))
			feature_dist_string += """<div id="SVM" class="w3-container feature_dist"  style='""" + display_str + """'><img src="data:image/png;base64,""" + b64_str + """\" style="width:50%;" class="border"></div>"""
		
			with open(path+"/feature_evaluation/SVM_tree_scores.png", "rb") as image_file:
				b64_str = str(base64.b64encode(image_file.read()).decode('utf-8').replace('\n', ''))
			feature_dist_string += """<div id="SVM_TREE" class="w3-container feature_dist" style="display:none;"><img src="data:image/png;base64,""" + b64_str + """\" style="width:50%;" class="border"></div>"""
			display_str = "display:none;"
	
		if m == "LASSO" and num_class==2:
			with open(path+"/feature_evaluation/LASSO_scores.png", "rb") as image_file:
				b64_str = str(base64.b64encode(image_file.read()).decode('utf-8').replace('\n', ''))
			feature_dist_string += """<div id="LASSO" class="w3-container feature_dist"  style='""" + display_str + """'><img src="data:image/png;base64,""" + b64_str + """\" style="width:50%;" class="border"></div>"""

			with open(path+"/feature_evaluation/LASSO_tree_scores.png", "rb") as image_file:
				b64_str = str(base64.b64encode(image_file.read()).decode('utf-8').replace('\n', ''))
			feature_dist_string += """<div id="LASSO_TREE" class="w3-container feature_dist" style="display:none;"><img src="data:image/png;base64,""" + b64_str + """\" style="width:50%;" class="border"></div>"""
			display_str = "display:none;"

		if m == "MLPNN":
			feature_dist_string += """<div id="MLPNN" class="w3-container feature_dist"  style="width:80vw;""" + display_str + """">"""
			for l in label_set:
				with open(path+"/feature_evaluation/MLPNN_" + str(l) + "_scores.png", "rb") as image_file:
					b64_str = str(base64.b64encode(image_file.read()).decode('utf-8').replace('\n', ''))
				feature_dist_string += """<img src="data:image/png;base64,""" + b64_str + """\" style="width:50%;float:left;margin-right:2px;" class="border">"""
			feature_dist_string += """</div>"""
			
			feature_dist_string += """<div id="MLPNN_TREE" class="w3-container feature_dist"  style="width:80vw;display:none;">"""
			for l in label_set:
				with open(path+"/feature_evaluation/MLPNN_tree_" + str(l) + "_scores.png", "rb") as image_file:
					b64_str = str(base64.b64encode(image_file.read()).decode('utf-8').replace('\n', ''))
				feature_dist_string += """<img src="data:image/png;base64,""" + b64_str + """\" style="width:50%;float:left;margin-right:2px;" class="border">"""
			feature_dist_string += """</div>"""		
			display_str = "display:none;"
		
		if m == "CNN":
			feature_dist_string += """<div id="CNN" class="w3-container feature_dist"  style="width:80vw;""" + display_str + """">"""
			for l in label_set:
				with open(path+"/feature_evaluation/CNN_" + str(l) + "_scores.png", "rb") as image_file:
					b64_str = str(base64.b64encode(image_file.read()).decode('utf-8').replace('\n', ''))
				feature_dist_string += """<img src="data:image/png;base64,""" + b64_str + """\" style="width:50%;float:left;margin-right:2px;" class="border">"""
			feature_dist_string += """</div>"""	
			




			
	feature_list = """<div style="width:80%;padding-bottom:10px;"><button class="button" onclick="openFeat('Raw')">Raw Features</button><button class="button" onclick="openFeat('Tree')">Tree Features</button></div>"""
	
	feature_list += """<div style="width:100%" id="Raw" class="w3-container feature_list"><table style="width:100%" class="taxonomy"><tr><th align="left" style="width:25%" height="50">Species</th><th style="width:10%" align="left">% in top-k</th><th style="width:15%" align="left">Wilcoxon rank</th><th style="width:20%" align="left">Wilcoxon p-value</th><th style="width:20%" align="left">Enriched class</tr>"""
	
	for f in ranking_set:
		feat = f.split("g__")[-1].split("s__")[-1]
		feature_list += """<tr><td class="microbe" style="font-style:italic;"><font color="darkblue">""" + feat.replace("noname", "").replace("_"," ") + """</td><td>""" + rank_dict[f]["percent"] + """</td><td>""" + rank_dict[f]["Wilcoxon_rank"] + """</td><td>""" + rank_dict[f]["Wilcoxon_p"] + """</td><td>""" + rank_dict[f]["Enriched"] + "</td></tr>"""
			
	feature_list += """</table></div>"""
	
	keys = tree_ranking_set.keys()
	
	key_list = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
	
	feature_list += """<div id="Tree" class="w3-container feature_list" style="display:none;">"""
	
	for key in key_list:
		if key in keys:
			if key == "species":
				font_str = """style="font-style:italic;" """
			else:
				font_str = ""
			if len(tree_ranking_set[key].values()) > 0:
				feature_list += """<table style="width:100%;margin-bottom:10px;" class="taxonomy"><tr><th align="left" style="width:25%" height="50">""" + key.title() + """</th><th style="width:10%" align="left">% in top-k</th><th style="width:15%" align="left">Wilcoxon rank</th><th style="width:20%" align="left">Wilcoxon p-value</th><th style="width:20%" align="left">Enriched class</tr>"""
				for f in sorted(tree_ranking_set[key].items(), key=lambda kv: kv[1], reverse=True):
					feat = f[0]
					feature_list += """<tr><td class="microbe" style="font-style:italic;"><font color="darkblue">""" + f[0].replace("noname", "").replace("_"," ") + """</td><td>""" + tree_rank_dict[feat]["percent"] + """</td><td>""" + tree_rank_dict[feat]["Wilcoxon_rank"] + """</td><td>""" + tree_rank_dict[feat]["Wilcoxon_p"] + """</td><td>""" + tree_rank_dict[feat]["Enriched"] + "</td></tr>"""
											
				feature_list += """</table>"""
	feature_list += """</div>"""
		



	message = """
	<body>
		<div class="svg-container">
			<svg viewbox="0 0 800 400" class="svg">
				<path id="curve" fill="darkblue" d="M 800 100 Q 400 150 0 100 L 0 0 L 800 0 L 800 300 Z">></path>
			</svg>
		</div>

		<header>
			<h1>Meta-Signer</h1>
		</header>

		<main>
			<h2>Run Settings</h2><br/>
            <table>
            <tr><td class="category" style="width:300px">Dataset</td><td class="descriptor">""" + str(dataset) + """</td></tr>
            <tr><td class="category" style="width:300px"> Date </td><td class="descriptor">""" + str(date) + """</td></tr>
	    <tr><td>&nbsp;</td><td>&nbsp;</td></tr>
            <tr><td class="category" style="width:300px">Filter Threshold</td><td class="descriptor">""" + str(filter_thresh) + """</td></tr>
	    <tr><td class="category" style="width:300px">Models Trained</td><td class="descriptor">""" + model_string + """</td></tr>
            <tr><td class="category" style="width:300px">Number of CV Splits</td><td class="descriptor">""" + str(test_splits) + """</td></tr>
            <tr><td class="category" style="width:300px">Number of Runs</td><td class="descriptor">""" + str(num_runs) + """</td></tr>
            </table>
            <br/><br/><br/><br/><br/>
			<h2>Model Evaluation</h2>
			""" + results_table_string + """
			<br/>
			<div style=width:80%;padding-bottom:10px;><table><tr>""" + auc_boxplot_button + """
			  <td><button class="button" onclick="openBoxplot('MCC')">MCC</button></td>
			  <td><button class="button" onclick="openBoxplot('Precision')">Precision</button></td>
			  <td><button class="button" onclick="openBoxplot('Recall')">Recall</button></td>
			  <td><button class="button" onclick="openBoxplot('F1')">F1</button></td>
			</tr></table></div>""" + auc_boxplot_str + """
				
			<div id="MCC" class="w3-container boxplot" """ + mcc_display + """>
				<img src="data:image/png;base64,""" + mcc_bp_b64_str + """\" style="width:80%" class="border">
			</div>

			<div id="Precision" class="w3-container boxplot" style="display:none">
				<img src="data:image/png;base64,""" + pre_bp_b64_str + """\" style="width:80%" class="border">
			</div>

			<div id="Recall" class="w3-container boxplot" style="display:none">
				<img src="data:image/png;base64,""" + rec_bp_b64_str + """\" style="width:80%" class="border">
			</div>
			
			<div id="F1" class="w3-container boxplot" style="display:none">
				<img  src="data:image/png;base64,""" + f1_bp_b64_str + """\" style="width:80%" class="border">
			</div>
			<br/><br/><br/><br/><br/>
            <h2>Features Extracted</h2><br/>
			<h3>Distribution of Feature Scores</h3>
			""" + feature_dist_string + """<br style="clear:both" /><br/>
			<h3>Feature Lists</h3>
			<br/>
			""" + feature_list + """
			
		</main>

		<footer>
			<p>Footer</p>
		</footer>
	</body>

	<style>
	@import 'https://fonts.googleapis.com/css?family=Ubuntu:300, 400, 500, 700';

	*, *:after, *:before {
	  margin: 0;
	  padding: 0;
	}

	.svg-container {
	  position: absolute;
	  top: 0;
	  right: 0;
	  left: 0;
	  z-index: -1;
	}

	svg {
	  path {
		transition: .1s;
	  }

	  &:hover path {
		d: path("M 800 300 Q 400 250 0 300 L 0 0 L 800 0 L 800 300 Z");
	  }
	}

	body {
	  background: #fff;
	  color: #333;
	  font-family: 'Ubuntu', sans-serif;
	  position: relative;
	}

	h1 {
	  font-size: 160;
	}

	h2 {
	  font-size: 64;
	  margin-bottom:5px;
	}

	h3 {
	  font-size: 32;
	}

	header {
	  color: #fff;
	  padding-top: 10vs;
	  padding-bottom: 10vw;
	  text-align: center;
	}

	.category {
	  font-size: 24;	
	  font-weight: bold;
	}
	
	.descriptor {
	  font-size: 24;
	}
	
	table.model_table {
	  font-size: 18;
	  text-align: center;
	  margin-top: 2vw;
	  margin-bottom: 50px;
	  height: 400px;
	  width: 50%;
	  float:left;
	}
	
	main {
	  background: linear-gradient(to bottom, #ffffff 0%, #dddee1 100%);
	  border-bottom: 1px solid rgba(0, 0, 0, .2);
	  padding: 10vh 0 80vh 2vw;
	  position: relative;
	  text-align: left;
	  overflow: hidden;
	  
	  &::after {
		border-right: 2px dashed #eee;
		content: '';
		position: absolute;
		top: calc(10vh + 1.618em);
		bottom: 0;
		left: 50%;
		width: 2px;
		height: 100%;
	  }
	}

	footer {
	  background: #dddee1;
	  padding: 5vh 0;
	  text-align: center;
	  position: relative;
	}

	a:link, a:visited {
	  color: white;
	  padding: 14px 25px;
	  text-decoration: none;
	  display: inline-block;
	}
	.button {
	  background-color: darkblue;
	  border: none;
	  color: white;
	  padding: 15px 32px;
	  text-align: center;
	  text-decoration: none;
	  display: inline-block;
	  font-size: 16px;
	  margin: 2px;
	}
	.border {
	  border-width:2px;
	  border-style:solid;
	  border-color:darkblue;
	}
	
	.microbe {
	  font-size:20;
	}
	.taxonomy {
	  font-size:24;
	  float:left;
	}
	.feature_dist {
	  width:80vw;
	}

	</style>
	
	<script>
	function openBoxplot(stat) {
	  var i;
	  var x = document.getElementsByClassName("boxplot");
	  for (i = 0; i < x.length; i++) {
		x[i].style.display = "none";  
	  }
	  document.getElementById(stat).style.display = "block";  
	}
	
	function openDist(stat) {
	  var i;
	  var x = document.getElementsByClassName("feature_dist");
	  for (i = 0; i < x.length; i++) {
		x[i].style.display = "none";  
	  }
	  document.getElementById(stat).style.display = "block";  
	}
	
	function openFeat(stat) {
	  var i;
	  var x = document.getElementsByClassName("feature_list");
	  for (i = 0; i < x.length; i++) {
		x[i].style.display = "none";  
	  }
	  document.getElementById(stat).style.display = "block";  
	}
	</script>
	"""
	html_file.write(message)
	html_file.close()
