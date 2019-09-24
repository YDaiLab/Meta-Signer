#####################################################################################
# Global settings
#####################################################################################

[Evaluation]

# NumberTestSplits	Number of partitions (k) in k-fold cross-validation [integer]
# NumberRuns		Number of iterations to run cross-validation [integer]
# Normalization		Normalization type [Standard or MinMax]
# Dataset		Dataset located in ../data
# FilterThresh		Filter OTUs based on porportion found in samples [float]
# TopK			Number of features in final list [integer]

NumberTestSplits = 10	
NumberRuns = 1	
Normalization = Standard
DataSet = Cirrhosis
FilterThresh = 0.1
TopK = 20

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
# GridCV		Number of partitions (k) to use in cross-validation for parameter selection [integer]

Train = True
GridCV = 5


#####################################################################################
# LASSO settings
#####################################################################################

[LASSO]

# Train			Turn LASSO models on or off [boolean]
# NumberIterations	Max iterations for LASSO training [integer]
# GridCV		Number of partitions (k) to use in cross-validation for parameter selection [integer]

Train = True
NumberIterations = 10000
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

#####################################################################################
# PopPhy-CNN models
#####################################################################################

[PopPhy]

# Train			Turn PopPhy-CNN models on or off [boolean]
# LearningRate		Learning rate for PopPhy-CNN models [float]
# BatchSize		Batch size for PopPhy-CNN models [integer]
# ValidationModels	Number of partitions (k) to use in cross-validation for network tuning [integer]
# Patience		Patience to use for early stopping for parameter tuning [integer]

Train = True
LearningRate = 0.001
BatchSize = 1024
Patience = 40

