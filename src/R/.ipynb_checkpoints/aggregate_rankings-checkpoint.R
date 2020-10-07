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
method <- myArgs[6]

input <- paste(path, "/feature_evaluation/ensemble_rank_table.csv", sep="")
output <- paste(path, "/feature_evaluation/aggregated_rank_table.csv", sep="")

scores <- paste(path, "/prediction_evaluation/results_AUC.tsv", sep="")

metric_scores <- as.data.frame(read.table(scores, sep="\t", header=T, colClasses="character", row.names=1))

rf_scores <- as.vector(metric_scores["RF",])
svm_scores <- as.vector(metric_scores["SVM",])
lasso_scores <- as.vector(metric_scores["Logistic Regression",])
mlpnn_scores <- as.vector(metric_scores["MLPNN",])
w <- c(rf_scores, svm_scores, lasso_scores, mlpnn_scores)


w <- as.numeric(w)
w <- w[!is.na(w)]
w[w<0] <- 0



mat <- read.table(input, sep=",", header=T, colClasses="character", row.names=1)
print(t(as.matrix(mat)))

if ((nrow(t(as.matrix(mat)))) == 1){
    ranking <- t(as.matrix(mat))[1:k]
}else
{
    ranking <- RankAggreg(t(as.matrix(mat)), k, importance=w, method=method, distance="Spearman", verbose=F)$top.list
}
write.table(ranking, output, sep=",", row.names=F, col.names=F, quote=F)
