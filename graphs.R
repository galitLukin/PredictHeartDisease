library(pROC)

data <- read.table("measures.csv", header=TRUE, sep=",")

labs = data[3:24,]
labstrain = labs[labs$trainORtest == "train", ] 
labstest = labs[labs$trainORtest == "test", ] 

plot(labstrain$param,labstrain$accuracy, type="b", col="blue", lwd=5, pch=15, xlab="Rho", ylab="Accuracy")
lines(labstest$param,labstest$accuracy, type="b", col="red", lwd=2, pch=19)
title("X Uncertainty Accuracy")
legend(0,2.8,c("train","test"), lwd=c(5,2), col=c("blue","red"), pch=c(15,19), y.intersp=1.5)

plot(labstrain$param,labstrain$precision, type="b", col="blue", lwd=5, pch=15, xlab="Rho", ylab="Precision",ylim=range(0.24,0.33))
lines(labstest$param,labstest$precision, type="b", col="red", lwd=2, pch=19)
title("X Uncertainty Precision")
legend(0,2.8,c("train","test"), lwd=c(5,2), col=c("blue","red"), pch=c(15,19), y.intersp=1.5)

plot(labstrain$param,labstrain$recall, type="b", col="blue", lwd=5, pch=15, xlab="Rho", ylab="Recall",ylim=range(0.35,0.7))
lines(labstest$param,labstest$recall, type="b", col="red", lwd=2, pch=19)
title("X Uncertainty Recall")
legend(0,2.8,c("train","test"), lwd=c(5,2), col=c("blue","red"), pch=c(15,19), y.intersp=1.5)

params = c("0.0", "0.5", "1.0", "1.5", "2.0", "2.5", "3.0", "3.5","4.0","4.5","5.0")
paramsI = c(0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5,4.0,4.5,5.0)
aucs = c()
for(i in params){
  aucd <- read.table(paste0("auc_trainlabErrors_",i,".csv"), header=TRUE, sep=",")
  roc_obj <- roc(aucd$y,aucd$prediction)
  x = auc(roc_obj)
  print(x[1])
  aucs <- c(aucs,x[1])
}
plot(paramsI,aucs, type="b", col="blue", lwd=5, pch=15, xlab="Rho", ylab="AUC",ylim=range(0.6,0.7))

aucs = c()
for(i in params){
  aucd <- read.table(paste0("auc_testlabErrors_",i,".csv"), header=TRUE, sep=",")
  roc_obj <- roc(aucd$y,aucd$prediction)
  x = auc(roc_obj)
  print(x[1])
  aucs <- c(aucs,x[1])
}
lines(paramsI,aucs, type="b", col="red", lwd=2, pch=19)
title("X Uncertainty AUC")
legend(0,2.8,c("train","test"), lwd=c(5,2), col=c("blue","red"), pch=c(15,19), y.intersp=1.5)



labels = data[25:46,]
labelstrain = labels[labels$trainORtest == "train", ] 
labelstest = labels[labels$trainORtest == "test", ] 
plot(labelstrain$param,labelstrain$accuracy, type="b", col="blue", lwd=5, pch=15, xlab="Gamma", ylab="Accuracy",ylim=range(0.55,0.7))
lines(labelstest$param,labelstest$accuracy, type="b", col="red", lwd=2, pch=19)
title("Y Uncertainty Accuracy")
legend(0,2.8,c("train","test"), lwd=c(5,2), col=c("blue","red"), pch=c(15,19), y.intersp=1.5)

plot(labelstrain$param,labelstrain$precision, type="b", col="blue", lwd=5, pch=15, xlab="Gamma", ylab="Precision",ylim=range(0.54,0.7))
lines(labelstest$param,labelstest$precision, type="b", col="red", lwd=2, pch=19)
title("Y Uncertainty Precision")
legend(0,2.8,c("train","test"), lwd=c(5,2), col=c("blue","red"), pch=c(15,19), y.intersp=1.5)

plot(labelstrain$param,labelstrain$recall, type="b", col="blue", lwd=5, pch=15, xlab="Gamma", ylab="Recall",ylim=range(0.35,0.7))
lines(labelstest$param,labelstest$recall, type="b", col="red", lwd=2, pch=19)
title("Y Uncertainty Recall")
legend(0,2.8,c("train","test"), lwd=c(5,2), col=c("blue","red"), pch=c(15,19), y.intersp=1.5)


params = c("0.0", "0.01", "0.02", "0.03", "0.04", "0.05", "0.06", "0.07", "0.08", "0.09", "0.1")
paramsI = c(0.0, 0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1)
aucs = c()
for(i in params){
  aucd <- read.table(paste0("auc_trainlabelError_",i,".csv"), header=TRUE, sep=",")
  roc_obj <- roc(aucd$y,aucd$prediction)
  x = auc(roc_obj)
  print(x[1])
  aucs <- c(aucs,x[1])
}
plot(paramsI,aucs, type="b", col="blue", lwd=5, pch=15, xlab="Gamma", ylab="AUC",ylim=range(0.5,0.7))

aucs = c()
for(i in params){
  aucd <- read.table(paste0("auc_testlabelError_",i,".csv"), header=TRUE, sep=",")
  roc_obj <- roc(aucd$y,aucd$prediction)
  x = auc(roc_obj)
  print(x[1])
  aucs <- c(aucs,x[1])
}
lines(paramsI,aucs, type="b", col="red", lwd=2, pch=19)
title("Y Uncertainty AUC")
legend(0,2.8,c("train","test"), lwd=c(5,2), col=c("blue","red"), pch=c(15,19), y.intersp=1.5)



