# Meta-Signer

Meta-Signer is a machine learning aggregated approach for feature evaluation of metagenomic datasets. Random forest, support vector machines, LASSO, multi-layer neural networks, and our novel PopPhy-CNN machine learning frameworks are used to train and evaluate features in a cross-validated fashion. Features are then aggregated across models and partitions into a single ranked list of the top *k* features.

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
### Set configuration settings

Meta-Signer offers a flexible framework which can be customized in the configuration file.

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

```bash
cd src
python meta-signer.py
``` 

