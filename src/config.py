#####################################################################################
# Global settings
#####################################################################################

[Evaluation]

# NumberTestSplits	Number of partitions (k) in k-fold cross-validation [integer]
# NumberRuns		Number of iterations to run cross-validation [integer]
# Normalization		Normalization type [Standard or MinMax]
# Dataset		Dataset located in ../data
# FilterThresh		Filter OTUs based on porportion found in samples [float]
# MaxK			Max number of features in final list [integer]

NumberTestSplits = 10
NumberRuns = 10
Normalization = Standard
DataSet = PRISM_3
FilterThreshCount = 0.1
FilterThreshMean = 0.001
MaxK = 100
AggregateMethod = GA

#####################################################################################
# Random Forest settings
#####################################################################################

[RF]

# Train			Turn Random Forest models on or off [boolean]
# NumberTrees		Number of trees per forest [integer]
# ValidationModels	Number of partitions (k) to use in cross-validation for selecting number of features [integer]

Train = True
NumberTrees = 500
ValidationModels = 5


#####################################################################################
# SVM settings
#####################################################################################

[SVM]

# Train			Turn SVM models on or off [boolean]
# MaxIterations		Max iterations for SVM training [integer]
# GridCV		Number of partitions (k) to use in cross-validation for parameter selection [integer]

Train = True
MaxIterations = 10000
GridCV = 5


#####################################################################################
# Logistic Regression settings
#####################################################################################

[Logistic Regression]

# Train			Turn Logistic Regression models on or off [boolean]
# MaxIterations		Max iterations for Logistic Regression training [integer]
# GridCV		Number of partitions (k) to use in cross-validation for parameter selection [integer]

Train = True
MaxIterations = 10000
GridCV = 5


#####################################################################################
# MLPNN settings
#####################################################################################

[MLPNN]

# Train			Turn MLPNN models on or off [boolean]
# LearningRate		Learning rate for MLPNN models [float]
# BatchSize		Batch size for MLPNN models [integer]
# ValidationModels	Number of partitions (k) to use in cross-validation for network tuning [integer]
# Patience		Patience to use for early stopping for parameter tuning [integer]

Train = True
LearningRate = 0.001
BatchSize = 1024
Patience = 40



