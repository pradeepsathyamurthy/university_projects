# Setting the working directory
setwd("D:/Courses/CSC529 - Python/Case_Study1/Submission")

# Loading the file to R
data_raw_dummies = read.csv(file="data_treated.csv",header = TRUE)
# Removing the index generated out of pandas library
data_raw_dummies = data_raw_dummies[2:42]

########################################
# Data Splitting - 80 - 20 split
########################################

# Installing the necessary library package
library(caret)
# Creating a random index to split the data as 80 - 20%
idx = createDataPartition(data_raw_dummies$y_yes, p=.80, list=FALSE)
#print(idx)
# Using the index created to create a Training Data set - 131 observations created
train_data1 = data_raw_dummies[idx,]
# Using the index created to create a Testing Data set - 31 observations created
test_data1 = data_raw_dummies[-idx,]


# Data describe
table(train_data1$y_yes)
table(test_data1$y_yes)

#check classes distribution
prop.table(table(train_data1$y_yes))

# Building tree on training set
library(rpart)
treeimb <- rpart(y_yes ~ ., data = train_data1)
pred.treeimb <- predict(treeimb, newdata = test_data1)
roc.curve(test_data1$y_yes, pred.treeimb,plotit=TRUE, col='Red')

# DT evaluation
confusionMatrix(test_data1$y_yes,pred.treeimb)

install.packages("ROSE")
library(ROSE)
accuracy.meas(test_data1$y_yes, pred.treeimb)

# ROC coverage
roc.curve(test_data1$y_yes, pred.treeimb, plotit = T)

help("ovun.sample")
# Over sampling
data_balanced_over <- ovun.sample(y_yes ~ ., data = train_data1, method = "over",N = 63822)$data
table(data_balanced_over$y_yes)


# under sampling without replacement
data_balanced_under <- ovun.sample(y_yes ~ ., data = train_data1, method = "under", N = 8516, seed = 1)$data
table(data_balanced_under$y_yes)

# Doing both 
data_balanced_both <- ovun.sample(y_yes ~ ., data = train_data1, method = "both", p=0.5, N=8516, seed = 1)$data
table(data_balanced_both$y_yes)

# Applying ROSE method
data.rose <- ROSE(y_yes ~ ., data = train_data1, seed = 1)$data
table(data.rose$y_yes)

# Applying SMOTE method
install.packages("DMwR")
library(DMwR)
set.seed(1234)
help("SMOTE")
str(train_data1)
train_data2 = train_data1
train_data2$y_yes = as.factor(train_data2$y_yes)
data.smote <- SMOTE(y_yes~., train_data2, perc.over= 200, perc.under = 150)
table(data.smote$y_yes)


############################################################
# Model Creation - Classification Technique - Decision Tree
############################################################

# Now lets build a DT on each of the different sampling done
tree.rose <- rpart(y_yes ~ ., data = data.rose)
tree.over <- rpart(y_yes ~ ., data = data_balanced_over)
tree.under <- rpart(y_yes ~ ., data = data_balanced_under)
tree.both <- rpart(y_yes ~ ., data = data_balanced_both)
tree.smote <- rpart(y_yes ~ ., data = data.smote)

#make predictions on unseen data
pred.tree.rose <- predict(tree.rose, newdata = test_data1)
pred.tree.over <- predict(tree.over, newdata = test_data1)
pred.tree.under <- predict(tree.under, newdata = test_data1)
pred.tree.both <- predict(tree.both, newdata = test_data1)
pred.tree.smote <- predict(tree.smote, newdata = test_data1)


#AUC ROSE - 0.758
roc.curve(test_data1$y_yes, pred.tree.rose,plotit=TRUE, col='Red')
#AUC Oversampling - 0.855
roc.curve(test_data1$y_yes, pred.tree.over,plotit=TRUE,add.roc=TRUE, col='Green')
#AUC Undersampling - 0.839
roc.curve(test_data1$y_yes, pred.tree.under,plotit=TRUE,add.roc=TRUE, col='Blue')
#AUC Both - 0.858
roc.curve(test_data1$y_yes, pred.tree.both,plotit=TRUE,add.roc=TRUE, col='Purple')
#AUC SMOTE - 0.817
roc.curve(test_data1$y_yes, pred.tree.smote[,2],plotit=TRUE,add.roc=TRUE, col='Brown')

legend('bottomright', 
       legend=c('Rose', 'Over_Sampling', 'Under_Sampling', 'Both_Over_Under_Sampling_Rose','SMOTE'),
       col = c('red', 'Green', 'Blue', 'Purple','Brown'),
       lty = 1:4,
       cex = 0.3)


#########################################################
# Model Creation - Classification Technique - KNN
#########################################################

# train a KNN model with K = 5
# Now lets build a KNN on each of the different sampling done
help("knn3")
knn.rose <- knn3(y_yes ~ ., data = data.rose, k=5)
knn.over <- knn3(y_yes ~ ., data = data_balanced_over, k=5)
knn.under <- knn3(y_yes ~ ., data = data_balanced_under, k=5)
knn.both <- knn3(y_yes ~ ., data = data_balanced_both, k=5)
knn.smote <- knn3(y_yes ~ ., data = data.smote, k=5)


#make predictions on unseen data
pred.knn.rose <- predict(knn.rose, newdata = test_data1)
pred.knn.over <- predict(knn.over, newdata = test_data1)
pred.knn.under <- predict(knn.under, newdata = test_data1)
pred.knn.both <- predict(knn.both, newdata = test_data1)
pred.knn.smote <- predict(knn.smote, newdata = test_data1)

#AUC ROSE - 0.662
roc.curve(test_data1$y_yes, pred.knn.rose[,1],plotit=TRUE, col='Red')
#AUC Oversampling - 0.667
roc.curve(test_data1$y_yes, pred.knn.over[,1],plotit=TRUE,add.roc=TRUE, col='Green')
#AUC Undersampling - 0.704
roc.curve(test_data1$y_yes, pred.knn.under[,1],plotit=TRUE,add.roc=TRUE, col='Blue')
#AUC Both - 0.686
roc.curve(test_data1$y_yes, pred.knn.both[,1],plotit=TRUE,add.roc=TRUE, col='Purple')
#AUC Smote - 0.695
roc.curve(test_data1$y_yes, pred.knn.smote[,1],plotit=TRUE,add.roc=TRUE, col='Brown')

legend('bottomright', 
       legend=c('Rose', 'Over_Sampling', 'Under_Sampling', 'Both_Over_Under_Sampling_Rose','SMOTE'),
       col = c('red', 'Green', 'Blue', 'Purple','Brown'),
       lty = 1:4,
       cex = 0.3)


# We will work on sample data produced by SMOTE as it has a reliable ROC curv with both classifier
library(caret)
# Decision Tree using 5-fold cross validation (ACC = 0.72, Sen = 0.71, Spe = 0.80)
help("trainControl")
trainctrl <- trainControl(method = "cv", number = 5)

# Recording the time - starts
ptm <- proc.time()
model1_tree <- train(y_yes~., data = data.smote, method='rpart', trControl = trainctrl)
# Stop the clock
proc.time() - ptm

# Recording the time - starts
ptm <- proc.time()
pred_model1_tree <- predict(model1_tree, test_data1)
# Stop the clock
proc.time() - ptm

# Model evaluation
confusionMatrix(pred_model1_tree, test_data1$y_yes)
accuracy.meas(pred_model1_tree,test_data1$y_yes)

library(rattle)
fancyRpartPlot(model1_tree$finalModel)

# Variable ranking based on importance
model1_tree$finalModel$variable.importance
#duration_min_max_norm       contact_unknown      contact_cellular  balance_min_max_norm 
#3765.194048            693.614990            351.788572            158.543329 
#campaign_min_max_norm            housing_no           housing_yes                 pdays 
#117.341840             52.962845             52.962845             49.847384 
#poutcome_unknown              previous            day_z_norm          month_z_norm 
#49.785075             49.535838             31.972355             31.972355 
#age_z_norm 
#3.955343 


# Naive Bayes Method classifier
# Recording the time - starts
ptm <- proc.time()
model1_nb <- train(y_yes~., data = data.smote, method='nb', trControl = trainctrl)
# Stop the clock
proc.time() - ptm

# Recording the time - starts
ptm <- proc.time()
pred_model1_nb <- predict(model1_nb, test_data1)
# Stop the clock
proc.time() - ptm

# Model Evaluation
confusionMatrix(pred_model1_nb, test_data1$y_yes)
accuracy.meas(pred_model1_nb,test_data1$y_yes)


# KNN method classifier
# Recording the time - starts
ptm <- proc.time()
model1_knn <- train(y_yes~., data = data.smote, method='knn', trControl = trainctrl)
# Stop the clock
proc.time() - ptm

# Recording the time - starts
ptm <- proc.time()
pred_model1_knn <- predict(model1_knn, test_data1)
# Stop the clock
proc.time() - ptm

# Computing metrics
confusionMatrix(pred_model1_knn, test_data1$y_yes)
accuracy.meas(pred_model1_knn, test_data1$y_yes)


# Logistic Regression with 5 fold (ACC = 0.844, Sen = 0.85, Spe = 0.78)
# Recording the time - starts
ptm <- proc.time()
model2_logistic <- train(y_yes~., data = data.smote, method='glm', trControl = trainctrl)
# Stop the clock
proc.time() - ptm

# Recording the time - starts
ptm <- proc.time()
pred_moel2_logistic <- predict(model2_logistic, test_data1)
# Stop the clock
proc.time() - ptm

confusionMatrix(pred_moel2_logistic, test_data1$y_yes)
accuracy.meas(pred_moel2_logistic, test_data1$y_yes)

model2_logistic$finalModel


# Ensemble approach on decision tree - (ACC = 0.8654, Sen = 0.87, Spe = 0.81)
# Random Forest: training with 5-fold CV (takes time to train and find the best model)

# Recording the time - starts
ptm <- proc.time()
model3_rf <- train(y_yes~., data = data.smote, method='rf', trControl = trainctrl)
# Stop the clock
proc.time() - ptm

# Recording the time - starts
ptm <- proc.time()
pred_moel3_rf <- predict(model3_rf, test_data1)
# Stop the clock
proc.time() - ptm

confusionMatrix(pred_moel3_rf, test_data1$y_yes)
accuracy.meas(pred_moel3_rf, test_data1$y_yes)

plot(model3_rf$finalModel)
model3_rf$finalModel$importance

# Boosted logistic regression
# Recording the time - starts
ptm <- proc.time()
model4_boosted <- train(y_yes~., data = data.smote, method='LogitBoost')
# Stop the clock
proc.time() - ptm

# Recording the time - starts
ptm <- proc.time()
pred_moel4_boosted <- predict(model4_boosted, test_data1)
# Stop the clock
proc.time() - ptm

# Evaluation
confusionMatrix(pred_moel4_boosted, test_data1$y_yes)
accuracy.meas(pred_moel4_boosted, test_data1$y_yes)
model4_boosted$finalModel$Stump
model4_boosted$finalModel$Stump

# ROC curves of all models

library(ROCR)

tree.pred <- prediction(predict(model1_tree$finalModel, test_data1, type='prob')[,1], test_data1$y_yes)
tree.perf <- performance(tree.pred, "tpr", "fpr")
logit.pred <- prediction(predict(model2_logistic$finalModel, test_data1, type='response'), test_data1$y_yes)
logit.perf <- performance(logit.pred, "tpr", "fpr")
rf.pred <- prediction(as.numeric(predict(model3_rf$finalModel, test_data1)), as.numeric(test_data1$y_yes))
rf.perf <- performance(rf.pred, "tpr", "fpr")
bglm.pred <- prediction(as.numeric(predict(model4_boosted$finalModel, test_data1)), as.numeric(test_data1$y_yes))
bglm.perf <- performance(bglm.pred, "tpr", "fpr")

plot(tree.perf, col = 'red', lty=1)
plot(logit.perf, add = TRUE, col = 'blue', lty=2)
plot(rf.perf, add=TRUE, col='green', lty=3)
plot(bglm.perf, add=TRUE, col='purple', lty=4)
abline(0, 1, lty = 5)

legend('bottomright', 
       legend=c('Decision Tree', 'Logistic Regression', 'Random Forest', 'Boosted Logistic Regression'),
       col = c('red', 'blue', 'green', 'purple'),
       lty = 1:4,
       cex = 0.3)

# Writing data to csv file for exploration
write.csv(data.smote, file = "data_smote.csv")
write.csv(data.rose, file = "data_rose.csv")
write.csv(data_balanced_both, file = "data_balanced_both.csv")
write.csv(data_balanced_under, file = "data_balanced_under.csv")
write.csv(data_balanced_over, file = "data_balanced_over.csv")
