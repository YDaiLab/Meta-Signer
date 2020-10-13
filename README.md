![alt text](https://zenodo.org/badge/DOI/10.5281/zenodo.4077403.svg)
# Meta-Signer

Meta-Signer is a machine learning aggregated approach for feature evaluation of metagenomic datasets. Random forest, support vector machines, logistic regression, and multi-layer neural networks. Features are then aggregated across models and partitions into a single ranked list of the top *k* features.

## Execution:

We provide a python environment which can be imported using the [Conda](https://www.anaconda.com/distribution/) python package manager.

Deep learning models are built using [Tensorflow](https://www.tensorflow.org/). Meta-Signer was designed using **Tensorflow v1.14.0**.

To fully utilize GPUs for faster training of the deep learning models, users will need to be sure that both [CUDA](https://developer.nvidia.com/cuda-toolkit-archive) and [cuDNN](https://developer.nvidia.com/cudnn) are properly installed.

Other dependencies should be downloaded upon importing the provided environment.

### Clone Repository
```bash
git clone https://github.com/YDaiLab/Meta-Signer.git
cd Meta-Signer
```
### Import Conda Environment

```bash
conda env create -f meta-signer.yml
source activate meta-signer
``` 

### Meta-Signer's Required Input

To use Meta-Signer on a dataset, first create a directory in the _data_ folder. This directory requires two files:

|File|Description |
| --- | --- |
|abundance.tsv|A **tab separated** file where each row is a feature and each column is a sample. The first column should be the feature ID. There should be no header of sample IDs |
|labels.txt|A text file where each row is the sample class value. Rows should be in the same order as columns found in _abundance.tsv_ |

Examples can be found in the _PRISM_ and _PRISM_3_ datasets provided.

### Set configuration settings

Meta-Signer offers a flexible framework which can be customized in the configuration file. The configuration file offers the following parameters:

|Evaluation| |
| --- | --- |
|NumberTestSplits|Number of partitions for cross-validation|
|NumberRuns|Number of indepenendant iterations of cross-validation to run |
|Normalization|Normalization method applied to data (Standard or MinMax) |
|DataSet|Directory in _data_ directory to load data from |
|FilterThreshCount|Remove features who are present in fewer than the specified fraction of samples |
|FilterThreshMean|Remove features with a mean value less than the specified value |
|MaxK|The maximum number of features to generate in the rank aggregation |
|AggregateMethod|The method used for rank aggregation (GA or CE) |

|RF| |
| --- | --- |
|Train|Use Random Forest for feature ranking and aggregation|
|NumberTrees|Number of decision trees per forest|
|ValidationModels|Number of partitions for internal cross-validation for tuning|

|SVM| |
| --- | --- |
|Train|Use SVM for feature ranking and aggregation|
|MaxIterations|Maximum number of iterations to train|
|GridCV|Number of partitions for internal cross-validation for tuning|

|Logistic Regression| |
| --- | --- |
|Train|Use logistic regression for feature ranking and aggregation|
|MaxIterations|Maximum number of iterations to train|
|GridCV|Number of partitions for internal cross-validation for tuning|

|MLPNN| |
| --- | --- |
|Train|Use MLPNN for feature ranking and aggregation|
|LearningRate|Learning rate for neural network models|
|BatchSize|Size of each batch during neural network training|
|Patience|Number of epochs to stop training after no improvement|

### Run the Meta-Signer pipeline:

Once the configuration is set to desired values, generate the aggregated feature list using:

```bash
cd src
python generate_feature_ranking.py
``` 

Upon completion, Meta-Signer will generate a directory in the results folder with the same name as set to the _DataSet_ flag in the configuration file. This directory will contain important files of interest including:

|File|Description |
| --- | --- |
|training_performance.html|A portable HTML file showing cross-validated evaluation of ML methods|
|feature_evaluation/ensemble_rank_table.csv|ranked lists of features for each method and each cross-validated run|
|feature_evaluation/aggregated_rank_table.csv|Aggregated ranked list of features|
|prediction_evaluation/results.tsv|Results table for cross-validated evaluation of ML methods|

Once the features have been aggregated into a single ranked list, the user can decide on how many features to use for the final training of ML models. Meta-Signer can generate these final trained ML models using a user specified number of features using:

```bash
cd src
python generate_models.py <DataSet> <k>
``` 

Where _DataSet_ is the directory in the results folder to use and _k_ is the final number of features to use during training. Additionally, the models can be trained on an external datset using:

```bash
cd src
python generate_models.py <DataSet> <k> -e <ExternalDataSet>
``` 

Where _ExternalDataSet_ is a directory in the data folder with _abundance.tsv_ and _labels.txt_ files. 

Upon completion, Meta-Signer will create a directory within the dataset's results directory that will contain:

|File|Description |
| --- | --- |
|feature_ranking.html|A portable HTML file the ranked features up to the specified value of _k_|
|rf_model.pkl|The trained random forest model in pickle format|
|logistic_regression_model.pkl|The trained logistic regression model in pickle format|
|svm_model.pkl|The trained SVM model in pickle format|
|mlpnn.h5|The trained neural network model in H5 format|
|training_results.tsv|The performance of trained models on the training set|
|external_results.tsv|The performance of trained models on the external test set|
