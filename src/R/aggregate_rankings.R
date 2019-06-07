#specify the packages of interest
packages = c("RankAggreg")

#use this function to check if each package is on the local machine
#if a package is installed, it will be loaded
#if any are not, the missing package(s) will be installed and loaded
package.check <- lapply(packages, FUN = function(x) {
    if (!require(x, character.only = TRUE)) {
		install.packages(x, dependencies = TRUE, repos = "http://cran.us.r-project.org")}
	library(x, character.only = TRUE)
    }
)

myArgs <- commandArgs(trailingOnly=T)

path <- myArgs[1]
k <- as.numeric(myArgs[2])
method <- myArgs[3]
model <- myArgs[4]
metric <- myArgs[5]

input <- paste(path, "/feature_evaluation/", model, "_", method, "_rank_table.csv", sep="")
output <- paste(path, "/feature_evaluation/", model, "_", method, "_rank_table_aggregated.csv", sep="")

scores <- paste(path, "/prediction_evaluation/results_",metric,".tsv", sep="")

metric_scores <- as.data.frame(read.table(scores, sep="\t", header=T, colClasses="character", row.names=1))

if (method == "raw"){
	rf_scores <- as.vector(metric_scores["RF",])
	svm_scores <- as.vector(metric_scores["SVM",])
	lasso_scores <- as.vector(metric_scores["LASSO",])
	mlpnn_scores <- as.vector(metric_scores["MLPNN",])
	w <- c(rf_scores, svm_scores, lasso_scores, mlpnn_scores)
}


if (method == "tree"){
	rf_scores <- as.vector(metric_scores["RF_TREE",])
	svm_scores <- as.vector(metric_scores["SVM_TREE",])
	lasso_scores <- as.vector(metric_scores["LASSO_TREE",])
	mlpnn_scores <- as.vector(metric_scores["MLPNN_TREE",])
	cnn_scores <- as.vector(metric_scores["CNN",])
	w <- c(rf_scores, svm_scores, lasso_scores, mlpnn_scores, cnn_scores)
		
}
w <- as.numeric(w)
w <- w[!is.na(w)]
w[w<0] <- 0
print(w)



mat <- read.table(input, sep=",", header=T, colClasses="character", row.names=1)
print(dim(t(as.matrix(mat))))
print(nrow(t(as.matrix(mat))))

if ((nrow(t(as.matrix(mat)))) == 1){
	ranking <- t(as.matrix(mat))[1:k]
}else
{
	ranking <- RankAggreg(t(as.matrix(mat)), k, importance=w, method="CE", distance="Spearman", verbose=F)$top.list
}
write.table(ranking, output, sep=",", row.names=F, col.names=F, quote=F)
