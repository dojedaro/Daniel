library(mlbench)
library(tidyverse)
library(caret)
library(Boruta)
library(dplyr)
library(tidyverse) # Load Libraries
library(dplyr)
library(knitr)
library(corrplot)
library(glmnet)
library(mice)
library(ggplot2)
library(Amelia)
library(caret)
library(readr)
if("pacman" %in% rownames(installed.packages()) == FALSE) {install.packages("pacman")} # Check if you have universal installer package, install if not

pacman::p_load("caret","ROCR","lift","xgboost")

library(caTools)
if("pacman" %in% rownames(installed.packages()) == FALSE) {install.packages("pacman")} # Check if you have universal installer package, install if not

pacman::p_load("caret","ROCR","lift","randomForest") 

df <- read.csv(choose.files(),sep=",",header=TRUE) #### train


unique(lapply(df,class))

missing_values <- sort(sapply(df, function(x) { sum(is.na(x)) }), decreasing=TRUE) # I checked to see if there are any missing values
missing_values 



library(SuperLearner)
library(nnls)
library(MASS)

set.seed(123)

df<- subset(df,select=-c(Id))

sample<-sample.split(df$diabetes,SplitRatio=0.75)
train = subset(df, sample == TRUE)
test = subset(df, sample == FALSE)
as.data.frame(table(test$diabetes))


############ RF 

train$diabetes<-as.factor(train$diabetes)
test$diabetes<-as.factor(test$diabetes)



model_forest <- randomForest(diabetes~. , data=train,
                             type="classification",
                             importance=TRUE,
                             ntree = 500,           # hyperparameter: number of trees in the forest
                             ntry=100,  # hyperparameter: number of random columns to grow each tree
                             nodesize = 10,         # hyperparameter: min number of datapoints on the leaf of each tree
                             maxnodes = 200,
                             cutoff= c(0.6,0.4)
                             
                             
) 

plot(model_forest)  
print(model_forest)
varImpPlot(model_forest) 

forest_probabilities<-predict(model_forest,newdata=test)
#forest_probabilities
forest_probabilities <-as.factor(forest_probabilities)
as.data.frame(table(forest_probabilities))

confusionMatrix(forest_probabilities,test$diabetes)



forest_ROC_prediction <- prediction(as.numeric(forest_probabilities), as.numeric(test$diabetes))


forest_ROC <- ROCR::performance(forest_ROC_prediction,"tpr","fpr") 
plot(forest_ROC) #Plot ROC curve

AUC.tmp <- ROCR::performance(forest_ROC_prediction,"auc") 
forest_AUC <- as.numeric(AUC.tmp@y.values) 
forest_AUC




####################XGBoost

matrix_train <- model.matrix(diabetes~ ., data = train)[,-1]
matrix_test <- model.matrix(diabetes~ ., data = test)[,-1]

matrix_new <-model.matrix(diabetes~., data=score)[,-1]
XGboost_prediction<-predict(model_XGboost,newdata=matrix_new) 

write.csv(XGboost_prediction, file = " Kaggle 2.csv")

model_XGboost<-xgboost(data = data.matrix(matrix_train), 
                       label = as.numeric(as.character(train$diabetes)), 
                       eta = 0.1,       # hyperparameter: learning rate 
                       max_depth = 25,  # hyperparameter: size of a tree in each boosting iteration
                       nrounds =300,       # hyperparameter: number of boosting iterations  
                       objective = "binary:logistic",
                       lambda=1,
                       gamma=8,
                       min_child_weight=20,
                       subsample=0.80,
                       colsample_bytree=0.4
)


XGboost_prediction<-predict(model_XGboost,newdata=matrix_test) 

confusionMatrix(as.factor(ifelse(XGboost_prediction<0.6,0,1)),test$diabetes,positive="0")

   
XGboost_ROC_prediction <- prediction(XGboost_prediction, test$diabetes) 
XGboost_ROC_testing <- ROCR::performance(XGboost_ROC_prediction,"tpr","fpr")
plot(XGboost_ROC_testing)

auc.tmp <- ROCR::performance(XGboost_ROC_prediction,"auc") 
XGboost_auc_testing <- as.numeric(auc.tmp@y.values) 
XGboost_auc_testing 

#########################
sl_models <- c("SL.randomForest", "SL.xgboost", "SL.glm")

ensembled_model<- SuperLearner(Y= as.numeric(as.character(train$diabetes)),X=train, family=binomial(), SL.library = sl_models)

ensembled_predictions <- predict(ensembled_model, newdata =test, onlySL=T)

conv_predictions <- ifelse(ensembled_predictions$pred>0.6,1,0)


confusionMatrix(as.factor(conv_predictions),(as.factor(test$diabetes)),positive="0")



#################################################################################################### Performance

score <- read.csv(choose.files(),sep=",",header=TRUE) #### predictions

unique(lapply(score,class))

unique(lapply(score$diabetes,class))
score$diabetes<-as.factor((score$diabetes))
missing_values <- sort(sapply(score, function(x) { sum(is.na(x)) }), decreasing=TRUE) # I checked to see if there are any missing values
missing_values


score<- subset(score,select=-c(Id))


#final <- data.frame(score)

sl_models <- c("SL.randomForest", "SL.xgboost", "SL.glm")

ensembled_model<- SuperLearner(Y= as.numeric(as.character(train$diabetes)),X=train, family=binomial(), SL.library = sl_models)

ensembled_predictions_new <- predict(ensembled_model, newdata =score, onlySL=T)

conv_predictions <- ifelse(ensembled_predictions_new$pred>0.5,1,0)




write.csv(conv_predictions, file = " Kaggle 1.csv")

