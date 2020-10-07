import numpy as np
import os
import struct
import base64
from array import array as pyarray
from numpy import unique
import pandas as pd

def generate_html_performance(dataset, path, config, to_train, results_df, label_set):
    
    html_file = open(path + "/training_performance.html", "w")
    filter_thresh_count = config.get('Evaluation', 'FilterThreshCount')
    filter_thresh_mean = config.get('Evaluation', 'FilterThreshMean')
    test_splits = config.get('Evaluation', 'NumberTestSplits')
    agg_method = config.get('Evaluation', 'AggregateMethod')
    num_runs = config.get('Evaluation', 'NumberRuns')

    roc_b64_str = ""
    auc_bp_b64_str = ""
    mcc_bp_b64_str = ""
    rec_bp_b64_str = ""
    pre_bp_b64_str = ""
    f1_bp_b64_str = ""

    with open(path+"/prediction_evaluation/auc_boxplots.png", "rb") as image_file:
        auc_bp_b64_str = str(base64.b64encode(image_file.read()).decode('utf-8').replace('\n', ''))
    
    with open(path+"/prediction_evaluation/mcc_boxplots.png", "rb") as image_file:
        mcc_bp_b64_str = str(base64.b64encode(image_file.read()).decode('utf-8').replace('\n', ''))
    
    with open(path+"/prediction_evaluation/recall_boxplots.png", "rb") as image_file:
        rec_bp_b64_str = str(base64.b64encode(image_file.read()).decode('utf-8').replace('\n', ''))

    with open(path+"/prediction_evaluation/precision_boxplots.png", "rb") as image_file:
        pre_bp_b64_str = str(base64.b64encode(image_file.read()).decode('utf-8').replace('\n', ''))

    with open(path+"/prediction_evaluation/f1_boxplots.png", "rb") as image_file:
        f1_bp_b64_str = str(base64.b64encode(image_file.read()).decode('utf-8').replace('\n', ''))

    with open(path+"/feature_evaluation/feature_selection.png", "rb") as image_file:
        feat_sel_lp_b64_str = str(base64.b64encode(image_file.read()).decode('utf-8').replace('\n', ''))
        
    num_class = len(label_set)
        
    model_string = ""
    
    for m in to_train:
        model_string = model_string + m + "&emsp;"    
    
    
    
    auc_boxplot_button ="""<td><button class="button" onclick="openBoxplot('AUC')">AUC</button></td>"""
    auc_boxplot_str = """<div id="AUC" class="w3-container boxplot"><img src="data:image/png;base64,""" + auc_bp_b64_str + """\" style="width:60%" class="border"></div>"""
    mcc_display = """ style="display: none;" """

    
    
    
    
    
    
    
    #Distribution buttons
    
    feature_dist_string = "<div style=width:100%;padding-bottom:10px;><table><tr>"
    
    for m in to_train:
        if m == "RF":
            feature_dist_string += """<td><button class="button" onclick="openDist('RF')">RF</button></td>"""

        if m == "SVM":
            feature_dist_string += """<td><button class="button" onclick="openDist('SVM')">SVM</button></td>"""

        if m == "Logistic Regression":
            feature_dist_string += """<td><button class="button" onclick="openDist('Logistic Regression')">Logistic Regression</button></td>"""

        if m == "MLPNN":
            feature_dist_string += """<td><button class="button" onclick="openDist('MLPNN')">MLPNN</button></td>"""
    feature_dist_string += """</tr></table>"""

    
    
    
    #Distribution images
    
    display_str = ""
    for m in to_train:
    
        if m == "RF":
            with open(path+"/feature_evaluation/RF_scores.png", "rb") as image_file:
                b64_str = str(base64.b64encode(image_file.read()).decode('utf-8').replace('\n', ''))
            feature_dist_string += """<div id="RF" class="w3-container feature_dist" style='""" + display_str + """ '><img src="data:image/png;base64,""" + b64_str + """\" style="width:50%;""" + display_str + """" class="border"></div>"""
        display_str = "display: none;"
        
        if m == "SVM":
            with open(path+"/feature_evaluation/SVM_scores.png", "rb") as image_file:
                b64_str = str(base64.b64encode(image_file.read()).decode('utf-8').replace('\n', ''))
            feature_dist_string += """<div id="SVM" class="w3-container feature_dist"  style='""" + display_str + """'><img src="data:image/png;base64,""" + b64_str + """\" style="width:50%;" class="border"></div>"""

            display_str = "display: none;"

        if m == "Logistic Regression":
            with open(path+"/feature_evaluation/Logistic_Regression_scores.png", "rb") as image_file:
                b64_str = str(base64.b64encode(image_file.read()).decode('utf-8').replace('\n', ''))
            feature_dist_string += """<div id="Logistic Regression" class="w3-container feature_dist"  style='""" + display_str + """'><img src="data:image/png;base64,""" + b64_str + """\" style="width:50%;" class="border"></div>"""

            display_str = "display: none;"

        if m == "MLPNN":
            feature_dist_string += """<div id="MLPNN" class="w3-container feature_dist"  style="width:80vw;""" + display_str + """">"""
            with open(path+"/feature_evaluation/MLPNN_scores.png", "rb") as image_file:
                b64_str = str(base64.b64encode(image_file.read()).decode('utf-8').replace('\n', ''))
            feature_dist_string += """<img src="data:image/png;base64,""" + b64_str + """\" style="width:50%;float:left;margin-right:2px;" class="border"></div>"""

            display_str = "display: none;"


 

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
            <h3>Run Settings</h3><br/>
            <table>
            <tr><td class="category" style="width:300px">Dataset</td><td class="descriptor">""" + str(dataset) + """</td></tr>
        <tr><td>&nbsp;</td><td>&nbsp;</td></tr>
            <tr><td class="category" style="width:300px">Filter Threshold (Count)</td><td class="descriptor">""" + str(filter_thresh_count) + """</td></tr>
            <tr><td class="category" style="width:300px">Filter Threshold (Mean)</td><td class="descriptor">""" + str(filter_thresh_mean) + """</td></tr>
            <tr><td class="category" style="width:300px">Rank Aggregation Method</td><td class="descriptor">""" + str(agg_method) + """</td></tr>

        <tr><td class="category" style="width:300px">Models Trained</td><td class="descriptor">""" + model_string + """</td></tr>
            <tr><td class="category" style="width:300px">Number of CV Splits</td><td class="descriptor">""" + str(test_splits) + """</td></tr>
            <tr><td class="category" style="width:300px">Number of Runs</td><td class="descriptor">""" + str(num_runs) + """</td></tr>
            </table>
            <br/><br/>
            <h3>Model Evaluation</h3>
            <br/>
            <div style=width:60%;padding-bottom:10px;><table><tr>""" + auc_boxplot_button + """
              <td><button class="button" onclick="openBoxplot('MCC')">MCC</button></td>
              <td><button class="button" onclick="openBoxplot('Precision')">Precision</button></td>
              <td><button class="button" onclick="openBoxplot('Recall')">Recall</button></td>
              <td><button class="button" onclick="openBoxplot('F1')">F1</button></td>
            </tr></table></div>""" + auc_boxplot_str + """
                
            <div id="MCC" class="w3-container boxplot" """ + mcc_display + """>
                <img src="data:image/png;base64,""" + mcc_bp_b64_str + """\" style="width:60%" class="border">
            </div>

            <div id="Precision" class="w3-container boxplot" style="display: none">
                <img src="data:image/png;base64,""" + pre_bp_b64_str + """\" style="width:60%" class="border">
            </div>

            <div id="Recall" class="w3-container boxplot" style="display: none">
                <img src="data:image/png;base64,""" + rec_bp_b64_str + """\" style="width:60%" class="border">
            </div>
            
            <div id="F1" class="w3-container boxplot" style="display: none">
                <img  src="data:image/png;base64,""" + f1_bp_b64_str + """\" style="width:60%" class="border">
            </div>
            <br/><br/>
            <h3>Distribution of Feature Scores</h3>
            """ + feature_dist_string + """</div><br style="clear:both" /><br/><br/>
            <h3>Model Performance Using Selected Features</h3>
            <br/>
             <div id="FL_LP" class="w3-container">
             <img src="data:image/png;base64,"""+ feat_sel_lp_b64_str + """\" style="width:40%" class="border">
             </div>
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
      font-size: 64;
    }

    h2 {
      font-size: 24;
      margin-bottom:5px;
    }

    h3 {
      font-size: 24;
    }

    header {
      color: #fff;
      padding-top: 10vs;
      padding-bottom: 10vw;
      text-align: center;
    }

    .category {
      font-size: 18;
      font-weight: bold;
    }
    
    .descriptor {
      font-size: 18;
    }

    table.model_table {
      border: 1px solid #1C6EA4;
      background-color: #EEEEEE;
      text-align: left;
      border-collapse: collapse;
    }
    table.model_table td, table.model_table th {
      border: 1px solid #AAAAAA;
      padding: 3px 2px;
    }
    table.model_table tbody td {
      font-size: 20;
    }
    table.model_table tr:nth-child(even) {
      background: #D0E4F5;
    }
    table.model_table thead {
      background: #1C6EA4;
      background: -moz-linear-gradient(top, #5592bb 0%, #327cad 66%, #1C6EA4 100%);
      background: -webkit-linear-gradient(top, #5592bb 0%, #327cad 66%, #1C6EA4 100%);
      background: linear-gradient(to bottom, #5592bb 0%, #327cad 66%, #1C6EA4 100%);
      border-bottom: 2px solid #444444;
    }
    table.model_table thead th {
      font-size: 24;
      font-weight: bold;
      color: #FFFFFF;
      border-left: 2px solid #D0E4F5;
    }
    table.model_table thead th:first-child {
      border-left: none;
    }

    table.model_table tfoot {
      font-size: 16;
      font-weight: bold;
      color: #FFFFFF;
      background: #D0E4F5;
      background: -moz-linear-gradient(top, #dcebf7 0%, #d4e6f6 66%, #D0E4F5 100%);
      background: -webkit-linear-gradient(top, #dcebf7 0%, #d4e6f6 66%, #D0E4F5 100%);
      background: linear-gradient(to bottom, #dcebf7 0%, #d4e6f6 66%, #D0E4F5 100%);
      border-top: 2px solid #444444;
    }
    table.model_table tfoot td {
      font-size: 20;
    }
    table.model_table tfoot .links {
      text-align: right;
    }
    table.model_table tfoot .links a{
      display: inline-block;
      background: #1C6EA4;
      color: #FFFFFF;
      padding: 2px 8px;
      border-radius: 5px;
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
      margin-bottom:50px;
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
      var x = document.getElementsByClassName("model_feature_list");
      for (i = 0; i < x.length; i++) {
        x[i].style.display = "none";  
      }
      document.getElementById(stat).style.display = "block";  
    }
    
    
    
    </script>
    """
    html_file.write(message)
    html_file.close()

    
    
    
    
    
    
    

def generate_html_feature_lists(dataset, path, config, to_train, model_rankings, agg_ranking_set, train_results_df, external_results_df, label_set, has_external=False):
    
    html_file = open(path + "/feature_ranking.html", "w")
    num_class = len(label_set)


    # Internal CV Performance on subset of features

    model_string = ""
    results_sub_table_string = """<div style="margin-bottom:50px;"><table class="model_table"><tr><th style="width:150px;text-align:left">Model</th><th style="width:100px">AUC</th><th style="width:100px">MCC</th><th style="width:100px">Precision</th><th style="width:100px">Recall</th><th style="width:100px">F1</th></tr>"""
    for m in to_train:
        model_string = model_string + m + "&emsp;"
        mean_auc = train_results_df.loc["AUC"][m]
        mean_mcc = train_results_df.loc["MCC"][m]
        mean_prec = train_results_df.loc["Precision"][m]
        mean_recall = train_results_df.loc["Recall"][m]
        mean_f1 = train_results_df.loc["F1"][m]

        results_sub_table_string += """<tr><td style="text-align:left">""" + m + "</td><td>" + mean_auc + "</td><td>" + mean_mcc + "</td><td>" + mean_prec + "</td><td>" + mean_recall + "</td><td>" + mean_f1 + "</td></tr>"
            
    results_sub_table_string += "</table>"
    
    
    results_sub_external_table_string = ""

    # Internal CV Performance on subset of features
    if has_external == True:
        results_sub_external_table_string = "<h3>Evaluation On External Data</h3>"
        results_sub_external_table_string += """<div style="margin-bottom:50px;"><table class="model_table"><tr><th style="width:150px;text-align:left">Model</th><th style="width:100px">AUC</th><th style="width:100px">MCC</th><th style="width:100px">Precision</th><th style="width:100px">Recall</th><th style="width:100px">F1</th></tr>"""
        sub_model_external_string = ""
        for m in to_train:
            sub_model_external_string = sub_model_external_string + m + "&emsp;"
            mean_auc = external_results_df.loc["AUC"][m]
            mean_mcc = external_results_df.loc["MCC"][m]
            mean_prec = external_results_df.loc["Precision"][m]
            mean_recall = external_results_df.loc["Recall"][m]
            mean_f1 = external_results_df.loc["F1"][m]

            results_sub_external_table_string += """<tr><td style="text-align:left">""" + m + "</td><td>" + mean_auc + "</td><td>" + mean_mcc + "</td><td>" + mean_prec + "</td><td>" + mean_recall + "</td><td>" + mean_f1 + "</td></tr>"
            
        results_sub_external_table_string += "</table></div>"


    #Model Feature Lists Buttons
    display_str = ""
    
    model_feature_lists = "<div style=width:100%;padding-bottom:10px;><table><tr>"
    
    for m in model_rankings:
        if m == "RF":
            model_feature_lists += """<td><button class="button" onclick="openFeat('RF_features')">RF</button></td>"""

        if m == "SVM":
            model_feature_lists += """<td><button class="button" onclick="openFeat('SVM_features')">SVM</button></td>"""

        if m == "Logistic Regression":
            model_feature_lists += """<td><button class="button" onclick="openFeat('Logistic Regression_features')">Logistic Regression</button></td>"""

        if m == "MLPNN":
            model_feature_lists += """<td><button class="button" onclick="openFeat('MLPNN_features')">MLPNN</button></td>"""
    model_feature_lists += """</tr></table></div>"""


        
    for m in model_rankings:
        model_feature_lists += """<div style='width:100%; """ + display_str + """'  id='""" + m + """_features' class="w3-container model_feature_list"><table style="width:100%" class="taxonomy"><tr><th align="left" style="width:25%" height="50">Microbe</th><th style="width:10%" align="left">% in top-k</th></tr>"""
        for f, p in zip(model_rankings[m]["Features"], model_rankings[m]["Percent"]):
            feat = f[0].split("g__")[-1].split("s__")[-1]
            model_feature_lists += """<tr><td class="microbe" style="font-style:italic;"><font color="darkblue">""" + feat.replace("noname", "").replace("_"," ") + """</td><td>""" + p + """</td></tr>"""
        model_feature_lists += """</table></div>"""
        display_str = "display: none;"

         
        
        
    # Ensemble Feature List
    feature_list=""


    feature_list += """<div style="width:100%" id="Raw" class="w3-container feature_list"><table style="width:100%" class="taxonomy"><tr><th align="left" style="width:25%" height="50">Microbe</th><th style="width:10%" align="left">% in top-k</th><th style="width:20%" align="left">Elevated class</th><th style="width:15%" align="left">PERMANOVA rank</th><th style="width:20%" align="left">Adjusted p-value</th></tr>"""

    for f in agg_ranking_set:
        feat = f.split("g__")[-1].split("s__")[-1]
        feature_list += """<tr><td class="microbe" style="font-style:italic;"><font color="darkblue">""" + feat.replace("noname", "").replace("_"," ") + """</td><td>""" + agg_ranking_set[f]["percent"] + """</td><td>""" + agg_ranking_set[f]["Enriched"] + "</td><td>""" + agg_ranking_set[f]["PERMANOVA_rank"] + """</td><td>""" + agg_ranking_set[f]["PERMANOVA_p"] + """</td></tr>"""

    feature_list += """</table></div>"""



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
            <h3>Run Settings</h3><br/>
            <table>
            <tr><td class="category" style="width:300px">Dataset</td><td class="descriptor">""" + str(dataset) + """</td></tr>
        <tr><td>&nbsp;</td><td>&nbsp;</td></tr>
        <tr><td class="category" style="width:300px">Models Trained</td><td class="descriptor">""" + model_string + """</td></tr>
            </table></br></br>
            <h3>Model Feature Lists</h3>
            <br/>"""+ model_feature_lists + """
             </div>
            <br/>
            <h3>Aggregated Feature List</h3>
            <br/>
            """ + feature_list + """
            <h3>Evaluation Over Training Data</h3>
            """ + results_sub_table_string + """
            <br/><br/>
            """ + results_sub_external_table_string + """
            <br/><br/>
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
      font-size: 64;
    }

    h2 {
      font-size: 32;
      margin-bottom:5px;
    }

    h3 {
      font-size: 28;
    }

    header {
      color: #fff;
      padding-top: 10vs;
      padding-bottom: 10vw;
      text-align: center;
    }

    .category {
      font-size: 18;
      font-weight: bold;
    }
    
    .descriptor {
      font-size: 18;
    }

    table.model_table {
      border: 1px solid #1C6EA4;
      background-color: #EEEEEE;
      text-align: left;
      border-collapse: collapse;
    }
    table.model_table td, table.model_table th {
      border: 1px solid #AAAAAA;
      padding: 3px 2px;
    }
    table.model_table tbody td {
      font-size: 20;
    }
    table.model_table tr:nth-child(even) {
      background: #D0E4F5;
    }
    table.model_table thead {
      background: #1C6EA4;
      background: -moz-linear-gradient(top, #5592bb 0%, #327cad 66%, #1C6EA4 100%);
      background: -webkit-linear-gradient(top, #5592bb 0%, #327cad 66%, #1C6EA4 100%);
      background: linear-gradient(to bottom, #5592bb 0%, #327cad 66%, #1C6EA4 100%);
      border-bottom: 2px solid #444444;
    }
    table.model_table thead th {
      font-size: 24;
      font-weight: bold;
      color: #FFFFFF;
      border-left: 2px solid #D0E4F5;
    }
    table.model_table thead th:first-child {
      border-left: none;
    }

    table.model_table tfoot {
      font-size: 16;
      font-weight: bold;
      color: #FFFFFF;
      background: #D0E4F5;
      background: -moz-linear-gradient(top, #dcebf7 0%, #d4e6f6 66%, #D0E4F5 100%);
      background: -webkit-linear-gradient(top, #dcebf7 0%, #d4e6f6 66%, #D0E4F5 100%);
      background: linear-gradient(to bottom, #dcebf7 0%, #d4e6f6 66%, #D0E4F5 100%);
      border-top: 2px solid #444444;
    }
    table.model_table tfoot td {
      font-size: 20;
    }
    table.model_table tfoot .links {
      text-align: right;
    }
    table.model_table tfoot .links a{
      display: inline-block;
      background: #1C6EA4;
      color: #FFFFFF;
      padding: 2px 8px;
      border-radius: 5px;
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
      margin-bottom:50px;
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
      var x = document.getElementsByClassName("model_feature_list");
      for (i = 0; i < x.length; i++) {
        x[i].style.display = "none";  
      }
      document.getElementById(stat).style.display = "block";  
    }
    
    
    
    </script>
    """
    html_file.write(message)
    html_file.close()
