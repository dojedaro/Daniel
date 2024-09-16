
library(dplyr)
library(DMwR)
###############################################################################new hope


#############################################################################SMOTE 


remove(list = ls())

new_training <- read.csv(file.choose())

new_validation <- read.csv(file.choose())


new_training <- mutate(new_training, target=if_else(target==0,0,1))
as.data.frame(table(new_training$target))


new_validation <- mutate(new_validation,target=if_else(target==0,0,1))
as.data.frame(table(new_validation$target))

new_training$target <- as.factor(new_training$target)


new_balanced_data <- SMOTE(target ~., data= new_training, perc.over = 500, perc.under = 160)
as.data.frame(table(new_balanced_data$target))

new_validation$target<- as.factor(new_validation$target)

############################################################################# model forest 

library (caret)
if("pacman" %in% rownames(installed.packages()) == FALSE) {install.packages("pacman")} # Check if you have universal installer package, install if not

pacman::p_load("caret","ROCR","lift","randomForest") 

model_forest <- randomForest(target~. , data=new_balanced_data,
                             type="classification",
                             importance=TRUE,
                             ntree = 250,           # hyperparameter: number of trees in the forest
                             ntry=5,  # hyperparameter: number of random columns to grow each tree
                             nodesize = 60,         # hyperparameter: min number of datapoints on the leaf of each tree
                             maxnodes = 300,
                             cutoff = c(0.57, 0.43)
                             
) 

plot(model_forest)  
print(model_forest)
varImpPlot(model_forest) 

forest_probabilities<-predict(model_forest,newdata=new_validation)
#forest_probabilities
forest_probabilities <-as.factor(forest_probabilities)
as.data.frame(table(forest_probabilities))

confusionMatrix(forest_probabilities,new_validation$target)

forest_ROC_prediction <- prediction(as.numeric(forest_probabilities), as.numeric(new_validation$target))


forest_ROC <- performance(forest_ROC_prediction,"tpr","fpr") 
plot(forest_ROC) #Plot ROC curve


#################################################################################AUC 
AUC.tmp <- performance(forest_ROC_prediction,"auc") 
forest_AUC <- as.numeric(AUC.tmp@y.values) 
forest_AUC

################################################################################# Lift chart
plotLift(forest_probabilities,  new_validation$target, cumulative = TRUE, n.buckets = 10) # Plot Lift chart





########################## Model Forest 2


model_forest2 <- randomForest(target~. , data=new_balanced_data,
                             type="classification",
                             importance=TRUE,
                             ntree = 200,           # hyperparameter: number of trees in the forest
                             ntry=3,  # hyperparameter: number of random columns to grow each tree
                             nodesize = 50,         # hyperparameter: min number of datapoints on the leaf of each tree
                             maxnodes = 250,
                             cutoff = c(0.60, 0.40)
                             
) 

plot(model_forest2)  
print(model_forest2)
varImpPlot(model_forest2) 

forest_probabilities2<-predict(model_forest2,newdata=new_validation)
#forest_probabilities
forest_probabilities2 <-as.factor(forest_probabilities2)
as.data.frame(table(forest_probabilities2))

confusionMatrix(forest_probabilities2,new_validation$target)


forest_ROC_prediction2 <- prediction(as.numeric(forest_probabilities2), as.numeric(new_validation$target))


forest_ROC2 <- performance(forest_ROC_prediction2,"tpr","fpr") 
plot(forest_ROC2) #Plot ROC curve


#################################################################################AUC 
AUC.tmp2 <- performance(forest_ROC_prediction2,"auc") 
forest_AUC2 <- as.numeric(AUC.tmp2@y.values) 
forest_AUC2


#######################################################################################XG BOOST


if("pacman" %in% rownames(installed.packages()) == FALSE) {install.packages("pacman")} # Check if you have universal installer package, install if not

pacman::p_load("caret","ROCR","lift","xgboost") #Check, and if needed install the necessary packages




matrix_new_training <- model.matrix(target~ ., data = new_balanced_data)[,-1]
matrix_new_validation <- model.matrix(target~ ., data = new_validation)[,-1]



model_XGboost<-xgboost(data = data.matrix(matrix_new_training), 
                       label = as.numeric(as.character(new_balanced_data$target)), 
                       
                       max_depth = 11,  
                       nrounds =50,       
                       objective = "binary:logistic",
                       )
                    
                       



XGboost_prediction<-predict(model_XGboost,newdata=matrix_new_validation) 

confusionMatrix(as.factor(ifelse(XGboost_prediction>0.57,1,0)),new_validation$target,positive="1")


##XGboost_predictions<- ifelse(XGboost_prediction>0.57,1,0)

#################################################################################ROC Curve
XGboost_ROC_prediction <- prediction(XGboost_prediction, new_validation$target) 
XGboost_ROC_testing <- performance(XGboost_ROC_prediction,"tpr","fpr")
plot(XGboost_ROC_testing)

##################################################################################AUC
auc.tmp <- performance(XGboost_ROC_prediction,"auc") #Create AUC data
XGboost_auc_testing <- as.numeric(auc.tmp@y.values) #Calculate AUC
XGboost_auc_testing #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - so so, below 60% - not much value

#### Lift chart
plotLift(XGboost_prediction, new_validation$Target, cumulative = TRUE, n.buckets = 10) # Plot Lift chart




#########################3 XGBoost model 2




model_XGboost2<-xgboost(data = data.matrix(matrix_new_training), 
                        label = as.numeric(as.character(new_balanced_data$target)), 
                        
                        max_depth = 28,  
                        nrounds =100,       
                        objective = "binary:logistic",
)





XGboost_prediction2<-predict(model_XGboost2,newdata=matrix_new_validation) 

confusionMatrix(as.factor(ifelse(XGboost_prediction2>0.57,1,0)),new_validation$target,positive="1")

XGboost_predictions2<- ifelse(XGboost_prediction2>0.57,1,0)



XGboost_ROC_prediction2 <- prediction(XGboost_prediction2, new_validation$target) 
XGboost_ROC_testing2 <- performance(XGboost_ROC_prediction2,"tpr","fpr")
plot(XGboost_ROC_testing2)

##################################################################################AUC
auc.tmp2 <- performance(XGboost_ROC_prediction2,"auc") #Create AUC data
XGboost_auc_testing2 <- as.numeric(auc.tmp2@y.values) #Calculate AUC
XGboost_auc_testing2 #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - so so, below 60% - not much value



############################################################################ Ensemble model

install.packages("SuperLearner")
install.packages("nnls")
library(SuperLearner)
library(nnls)
library(MASS)

set.seed(123)



committee_pred = as.data.frame(cbind(as.character(forest_probabilities),as.character(forest_probabilities2), as.character(XGboost_predictions), as.character(XGboost_predictions2)))
head(committee_pred, n=77)
str(committee_pred)

committee_pred$one_count = apply(committee_pred[1:50006,1:4], 1, function(x) sum(x=="1"))    # (change 3 to the total number of predictions you have)
str(committee_pred$one_count)
committee_pred$zero_count = apply(committee_pred[1:50006,1:4], 1, function(x) sum(x=="0")) # (change 3 to the total number of predictions you have)
committee_pred$vote = factor(ifelse(committee_pred$one_count >= committee_pred$zero_count, "1", "0"))
head(committee_pred, n=77)

caret::confusionMatrix(data= as.factor(committee_pred$vote), reference= as.factor(new_validation$target), dnn=c("Predicted", "Actual"))




##SL.randomForest.tuning <- function(...){
###  SL.randomForest(..., model_forest)
#}
  
##SL.xgboost.tuning <- function(...){
 ## SL.xgboost(..., model_XGboost)
#} 


sl_models <- c("SL.randomForest", "SL.xgboost")
  
ensembled_model<- SuperLearner(Y= as.numeric(as.character(new_balanced_data$target)),X=new_balanced_data, family=binomial(), SL.library = sl_models)

ensembled_model



evaluation <- CV.SuperLearner(Y= as.numeric(as.character(new_balanced_data$target)),X=new_balanced_data, family=binomial(), SL.library= sl_models)
plot(evaluation)

ensembled_predictions <- predict(ensembled_model, newdata = new_validation, onlySL=T)

ensembled_predictions


conv_predictions <- ifelse(ensembled_predictions$pred>0.57,1,0)
conv_predictions


confusionMatrix(as.factor(conv_predictions),(as.factor(new_validation$target)),positive="0")


ensembled_model_ROC_prediction <- prediction(as.numeric(conv_predictions), as.numeric(new_validation$target))


ensembled_ROC <- performance(ensembled_model_ROC_prediction,"tpr","fpr") 
plot(ensembled_ROC)


AUC.tmp <- performance(ensembled_model_ROC_prediction,"auc") 
ensembled_AUC <- as.numeric(AUC.tmp@y.values) 
ensembled_AUC 



