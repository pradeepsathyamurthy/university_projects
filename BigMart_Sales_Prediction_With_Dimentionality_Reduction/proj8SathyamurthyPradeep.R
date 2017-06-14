######################################################################################
# Author: Pradeep Sathyamurthy
# Date: 07-June-2017
# Course: CSC-433
# Guiding Prof: Prof. Steve Jost
# Project: Final Project Submission
# Train Dataset Name: mart_train.csv
# Test Dataset Name: mart_test.csv
######################################################################################

# Libraries imported for this analysis
require(ggplot2) # <- needed for graphing
require(rpart) # <- Needed for building decision tree
require(rattle) # <- Needed to make decision tree look neat
require(rpart.plot) # <- Needed to make decision tree look neat
require(RColorBrewer) # <- Needed to make decision tree look neat
require(caret) # <- Needed for data splitting
require(MASS) # <- Needed for Outlier and Influential points detection 
require(car) # Needed for Multicolinearity

# Step-1: Reading the trianing dataset
setwd("C:/Users/prade/Documents/GitHub/university_projects/BigMart_Sales_Prediction_With_Dimentionality_Reduction")
data.mart.raw <- read.csv("Dataset/Mart_Train.csv")
head(data.mart.raw)

# Step-2: Researching the variables present
col_mart_name <- colnames(data.mart.raw) # <- Column names
col_mart_length <- length(col_mart_name) # <- There are 12 variables
var_det <- data.frame(Var_Name="NULL",Var_Type="NULL",stringsAsFactors = FALSE)
for(i in 1:col_mart_length){
    var_det <- rbind(var_det, c(colnames(data.mart.raw[i]),class(data.mart.raw[[i]])))
}
var_det <- var_det[-c(1),]
plot_var_type <- data.frame(table(var_det$Var_Type))
barplot(plot_var_type$Freq,names.arg = plot_var_type$Var1, main = "Variable Type Distribution in Dataset")
print(var_det,row.names = FALSE)
# above for loop says there are:
# 7 Factor Variables: Item_Identifier, Item_Fat_Content, Item_Type, Outlet_Identifier, Outlet_Size, Outlet_Location_Type, Outlet_Type
# 1 integer variable: Outlet_Establishment_Year
# 4 Numeric variables: Item_Weight, Item_Visibility, Item_MRP, Item_Outlet_Sales


# Step-3: Converting the object type based on their values
# From the data we could conclude to have Item_Identifier as a ID variable and Outlet_Establishment_Year as a factor
#data.mart.raw$Item_Identifier <- as.character(data.mart.raw$Item_Identifier)
data.mart.raw <- data.mart.raw[-c(1)]
head(data.mart.raw)
data.mart.raw$Outlet_Establishment_Year <- as.factor(data.mart.raw$Outlet_Establishment_Year)
summary(data.mart.raw) 
col_mart_name <- colnames(data.mart.raw) # <- Column names
col_mart_length <- length(col_mart_name) # <- There are 12 variables
var_det <- data.frame(Var_Name="NULL",Var_Type="NULL",stringsAsFactors = FALSE)
for(i in 1:col_mart_length){
    var_det <- rbind(var_det, c(colnames(data.mart.raw[i]),class(data.mart.raw[[i]])))
}
var_det <- var_det[-c(1),]
plot_var_type <- data.frame(table(var_det$Var_Type))
barplot(plot_var_type$Freq,names.arg = plot_var_type$Var1, main = "Variable Type Distribution in Dataset")
print(var_det,row.names = FALSE)


# Step-4: Exploratory Data Analysis on factor variables
# After conversion below are factor variables:
# 1. Item_Fat_Content
# 2. Item_Type
# 3. Outlet_Identifier
# 4. Outlet_Size
# 5. Outlet_Location_Type
# 6. Outlet_Type
# 7. Outlet_Establishment_Year
# Let us plot these data to see the frequency of occurence
data.frame(table(data.mart.raw$Item_Fat_Content))
plot(data.frame(table(data.mart.raw$Item_Fat_Content)), main="Frequency Distribution of Item_Fat_Content",xlab="Item_Fat_Content")
data.frame(table(data.mart.raw$Item_Type))
plot(data.frame(table(data.mart.raw$Item_Type)), main="Frequency Distribution of Item_Type",xlab="Item_Type")
data.frame(table(data.mart.raw$Outlet_Identifier))
plot(data.frame(table(data.mart.raw$Outlet_Identifier)), main="Frequency Distribution of Outlet_Identifier",xlab="Outlet_Identifier")
data.frame(table(data.mart.raw$Outlet_Size))
plot(data.frame(table(data.mart.raw$Outlet_Size)), main="Frequency Distribution of Outlet_Size",xlab="Outlet_Size")
data.frame(table(data.mart.raw$Outlet_Location_Type))
plot(data.frame(table(data.mart.raw$Outlet_Location_Type)), main="Frequency Distribution of Outlet_Location_Type",xlab="Outlet_Location_Type")
data.frame(table(data.mart.raw$Outlet_Type))
plot(data.frame(table(data.mart.raw$Outlet_Type)), main="Frequency Distribution of Outlet_Type",xlab="Outlet_Type")
data.frame(table(data.mart.raw$Outlet_Establishment_Year))
plot(data.frame(table(data.mart.raw$Outlet_Establishment_Year)), main="Frequency Distribution of Outlet_Establishment_Year",xlab="Outlet_Establishment_Year")


# Step-5: Exploratory Data Analysis on numerical variables
# After conversion below are numerical variables:
# 1. Item_Weight
# 2. Item_Visibility
# 3. Item_MRP
# 4. Item_Outlet_Sales
summary(data.mart.raw$Item_Weight)
hist(data.mart.raw$Item_Weight)
summary(data.mart.raw$Item_Visibility)
hist(data.mart.raw$Item_Visibility)
summary(data.mart.raw$Item_MRP)
hist(data.mart.raw$Item_MRP)
summary(data.mart.raw$Item_Outlet_Sales)
hist(data.mart.raw$Item_Outlet_Sales)
boxplot(data.mart.raw$Item_Outlet_Sales)


# Step-6: Treating the missing values
# From above exploratoy analysis, we could see there is no normal distriution of data in both factor as well numerical variable
# So before we normalize them, we need to treat missing values
head(data.mart.raw)
# Treating factor variables
pie(table((data.mart.raw$Item_Fat_Content)),main = "Analysis of Missing Values in Item_Fat_Content")
pie(table((data.mart.raw$Item_Type)),main = "Analysis of Missing Values in Item_Type")
pie(table((data.mart.raw$Outlet_Identifier)),main = "Analysis of Missing Values in Outlet_Identifier")
pie(table((data.mart.raw$Outlet_Establishment_Year)),main = "Analysis of Missing Values in Outlet_Establishment_Year")
pie(table((data.mart.raw$Outlet_Size)),main = "Analysis of Missing Values in Outlet_Size")
pie(table((data.mart.raw$Outlet_Location_Type)),main = "Analysis of Missing Values in Outlet_Location_Type")
pie(table((data.mart.raw$Outlet_Type)),main = "Analysis of Missing Values in Outlet_Type")
# Treating numerical variables
pie(table(is.na(data.mart.raw$Item_Weight)),main = "Analysis of Missing Values in Item_Weight")
pie(table(is.na(data.mart.raw$Item_Visibility)),main = "Analysis of Missing Values in Item_Visibility")
pie(table(is.na(data.mart.raw$Item_MRP)),main = "Analysis of Missing Values in Item_MRP")
pie(table(is.na(data.mart.raw$Item_Outlet_Sales)),main = "Analysis of Missing Values in Item_Outlet_Sales")

# Step-6.1: Treating Outlet_Size, Creating split based on the missing values in column Outlet_Size
data.mart.raw.tree <- data.mart.raw
data.mart.raw.tree.test <- data.mart.raw.tree[data.mart.raw.tree$Outlet_Size=="",]
data.mart.raw.tree.train <- data.mart.raw.tree[data.mart.raw.tree$Outlet_Size!="",]

# Step-6.2: Imputing values for outlet_size using decision tree
head(data.mart.raw.tree.train)
#tree_treated <- rpart(y~age+job+marital+education+default+balance+housing+loan+contact+day+month+duration+campaign+pdays+previous+poutcome,data=TRAINING_TREATEDBANKPROJECTDATASET)
tree_treated <- rpart(Outlet_Size~Item_Weight+Item_Fat_Content+Item_Visibility+Item_Type+Item_MRP+Outlet_Identifier+Outlet_Establishment_Year+Outlet_Location_Type+Outlet_Type+Item_Outlet_Sales, data = data.mart.raw.tree.train)
summary(tree_treated)
# Plotting the tree ( it is better though)
plot(tree_treated, uniform=TRUE)
# Now creating the fancy part
fancyRpartPlot(tree_treated)
# We can do prediction as below
predict(tree_treated)
predict(tree_treated, type="class")
# Confusion matrix
table(data.mart.raw.tree.train$Outlet_Size, predict(tree_treated, type="class"), dnn=c("Actual","Predicted"))
# Testing the model with test datpredicted_treated_class1a set
# Loading the file to R
predicted_treated_class <- predict(tree_treated,data.mart.raw.tree.test,type="class")
table(data.mart.raw.tree.test$Outlet_Size,predicted_treated_class,dnn=c("Actual","Predicted"))
# treating the missing values
for (i in 1 : length(data.mart.raw.tree.test$Outlet_Size)){
    if(data.mart.raw.tree.test$Outlet_Identifier[i] == ("OUT018") |
       data.mart.raw.tree.test$Outlet_Identifier[i] == ("OUT027") |
       data.mart.raw.tree.test$Outlet_Identifier[i] == ("OUT049")){
        data.mart.raw.tree.test$Outlet_Size[i] <- as.character("Medium")
    } else if (data.mart.raw.tree.test$Outlet_Identifier[i] == ("OUT013")){
        data.mart.raw.tree.test$Outlet_Size[i] <- as.character("High")
    } else {data.mart.raw.tree.test$Outlet_Size[i] <- as.character("Small")}
}
tail(data.mart.raw.tree.test$Outlet_Size)
data.mart.raw.tree <- rbind(data.mart.raw.tree.train,data.mart.raw.tree.test)
tail(data.mart.raw.tree)
data.mart.raw.2 <- data.mart.raw.tree

# Step:6.3 Treating Item_Weight
data.mart.raw.3 <- data.mart.raw.2
tail(data.mart.raw.3)
summary(data.mart.raw.3$Item_Weight) # <- from summary we see mean and median stay close, so i will fill data with its mean value
for (i in 1 : length(data.mart.raw.3$Item_Weight)){
    if(is.na(data.mart.raw.3$Item_Weight[i]) == TRUE |
       is.nan(data.mart.raw.3$Item_Weight[i]) == TRUE |
       is.null(data.mart.raw.3$Item_Weight[i]) == TRUE){
        data.mart.raw.3$Item_Weight[i] <- mean(data.mart.raw.3$Item_Weight, na.rm = TRUE)
    }
}
summary(data.mart.raw.3$Item_Weight) # <- From this we could see that mean and median became so close and hence we can hope this imputation works fine
data.mart.treaded <- data.mart.raw.3
hist(data.mart.treaded$Item_Weight) #<- Converted from normal curve

# Step:6.4 Treating Item_Weight Item_Fat_Content
data.frame(table(data.mart.treaded$Item_Fat_Content))
plot(data.frame(table(data.mart.treaded$Item_Fat_Content)), main="Frequency Distribution of Item_Fat_Content",xlab="Item_Fat_Content")
data.mart.treaded$Item_Fat_Content <- as.character(data.mart.treaded$Item_Fat_Content)
for (i in 1 : length(data.mart.treaded$Item_Fat_Content)){
    if(data.mart.treaded$Item_Fat_Content[i] == as.character("LF") |
       data.mart.treaded$Item_Fat_Content[i] == as.character("low fat") |
       data.mart.treaded$Item_Fat_Content[i] == as.character("Low Fat")){
        data.mart.treaded$Item_Fat_Content[i] <- as.character("Low_Fat")
    } else {data.mart.treaded$Item_Fat_Content[i] <- as.character("Regular")}
}

# Step:6.5 Converting the Column objects to factor or Numeric after treatment
data.mart.treaded$Item_Fat_Content <- as.factor(data.mart.treaded$Item_Fat_Content)
data.mart.treaded$Outlet_Size <- factor(data.mart.treaded$Outlet_Size,levels=c("High", "Medium", "Small"))

# Step:7 Splitting the dataset to test and train for local validation
# Creating a random index to split the data as 80 - 20%
idx <- createDataPartition(data.mart.treaded$Item_Weight, p=.80, list=FALSE)
print(idx[1:20])
# Using the index created to create a Training Data set - 131 observations created
data.train <- data.mart.treaded[idx,]
head(data.mart.treaded)
# Using the index created to create a Testing Data set - 31 observations created
data.test <- data.mart.treaded[-idx,]
head(data.test)
idx <- NULL

# Step-8 Exploratory data analysis on training set
# Factor Variables
data.frame(table(data.train$Item_Fat_Content))
plot(data.frame(table(data.train$Item_Fat_Content)), main="Frequency Distribution of Item_Fat_Content",xlab="Item_Fat_Content")
data.frame(table(data.train$Item_Type))
plot(data.frame(table(data.train$Item_Type)), main="Frequency Distribution of Item_Type",xlab="Item_Type")
data.frame(table(data.train$Outlet_Identifier))
plot(data.frame(table(data.train$Outlet_Identifier)), main="Frequency Distribution of Outlet_Identifier",xlab="Outlet_Identifier")
data.frame(table(data.train$Outlet_Size))
plot(data.frame(table(data.train$Outlet_Size)), main="Frequency Distribution of Outlet_Size",xlab="Outlet_Size")
data.frame(table(data.train$Outlet_Location_Type))
plot(data.frame(table(data.train$Outlet_Location_Type)), main="Frequency Distribution of Outlet_Location_Type",xlab="Outlet_Location_Type")
data.frame(table(data.train$Outlet_Type))
plot(data.frame(table(data.train$Outlet_Type)), main="Frequency Distribution of Outlet_Type",xlab="Outlet_Type")
data.frame(table(data.train$Outlet_Establishment_Year))
plot(data.frame(table(data.train$Outlet_Establishment_Year)), main="Frequency Distribution of Outlet_Establishment_Year",xlab="Outlet_Establishment_Year")
# Numerical Variabes
summary(data.train$Item_Weight)
hist(data.train$Item_Weight)
summary(data.train$Item_Visibility)
hist(data.train$Item_Visibility)
summary(data.train$Item_MRP)
hist(data.train$Item_MRP)
summary(data.train$Item_Outlet_Sales)
hist(data.train$Item_Outlet_Sales)
pie(table((data.train$Outlet_Size)),main = "Analysis of Missing Values in Outlet_Size")
pie(table(is.na(data.train$Item_Weight)),main = "Analysis of Missing Values in Item_Weight")

# Step-9 : Making Inference and Hypothesis
# 1. Low fat food is being purchased more compare to the regular fat foods
# 2. Food products like Fruits and Vegitables, snaks have higher sale; Households, canned, dairy and baking good have average sales and others are bought even less
# 3. OUT010 and OUT019 have lowest sale compare to others
# 4. Big mart owns Small and medium sized outlets more when comapre to High size outlet
# 5. Big mart outlets are situated more more in Tier3 and Tier2 locations when compare to Tier1 regions
# 6. Other than 1997, we could see a constant sale obtained in all years till 
# 7. Item weight has a normal distribution, which means product of all weight are available in store at equal proportion, it not just the whole sale which is happening in store
# 8. Product visibility is sckewed to right, stores have more of small display area for product more and interestingly there is a size 0 which can be even online sold product
# 9. MRP of the product is also quite normally distributed, which means product of all price range from $31 to $266 is available in store in eqal proportion, so it target all kind of customers for its sales
# 10. Total sale revenue is skewed to right, meaning store constantly generate revenue of range $800 to $3000 in each of its outlet mostly
# Hypothesis: Groceries like fruit, vegetables and snkacks with low fat content with minimum product visibility in a small and medium sized outlet situated in Tire-3 and Tier-2 region should have a comparitively good sale excluding the outlets OUT010 and OUT019.

# Step-10 : Basic Model Building
model1 <- lm(Item_Outlet_Sales~Item_Fat_Content+Item_Type+Outlet_Identifier+Outlet_Establishment_Year+Outlet_Size+Outlet_Location_Type+Outlet_Type+Item_Weight+Item_Visibility+Item_MRP,data = data.train)
cor_var1 <- data.frame(data.train$Item_Weight,data.train$Item_Visibility,data.train$Item_MRP)
cor(cor_var1) # No significant correlation exists with all numerical variabels available
summary(model1) # <- model-1 explains 0.5657 of sales variance, having Item_Fat_Content, Outlet_Identifier and Item_MRP as a significant variables
# Item_Outlet_Sales ~ Item_Fat_Content + Outlet_Identifier + Item_MRP

# Step-11 : Model Building using stepwise algorithm
model2_stepwise <- step(model1, direction = "backward")
summary(model2_stepwise) # <- explains 0.566 of sales variance
# Item_Outlet_Sales ~ Outlet_Identifier + Item_MRP

# Step-12: Residual Analysis
par(mfrow=c(4,2))
par(mar = rep(2, 4))
plot(model2_stepwise)
sd(data.train$Item_Outlet_Sales)
residual <- rstandard(model2_stepwise)
hist(residual) # Residual seems normally distributed
# Could observe some heteroscadastic behavious in residual plot, we can try for some transformation

# Step-13: Transformation
# Doing log transformation on dependent variable
model3_transformed <- lm(log(Item_Outlet_Sales)~Item_Fat_Content+Item_Type+Outlet_Identifier+Outlet_Establishment_Year+Outlet_Size+Outlet_Location_Type+Outlet_Type+Item_Weight+Item_Visibility+Item_MRP,data = data.train)
summary(model3_transformed)
# Adj R^2 is 0.7241
par(mfrow=c(4,2))
par(mar = rep(2, 4))
plot(model3_transformed)
residual_af_tranformation <- rstandard(model3_transformed)
hist(residual_af_tranformation)

# Step-14: Outlier Check and Influential Point Check
# computing studentized residual for outlier check
n_sample_size <- nrow(data.train)
studentized.residuals <- studres(model3_transformed)
#cat("Complete list of Studentized Residual::::","\n")
#print(studentized.residuals)
for(i in c(1:n_sample_size)){
    if(studentized.residuals[i] < -3 || studentized.residuals[i] > 3){
        cat("Validate these values for outliers:::",studentized.residuals[i],"at observation",i,"\n")
    }
}
# Influential Points
hhat.model <- lm.influence(model3_transformed)$hat
n_sample_size <- nrow(data.train)
p_beta <- length(model3_transformed$coefficients) +1
#cat("Complete list of HHat Values::::","\n")
#print(hhat.model)
hhat.cutoff <- (2*p_beta)/n_sample_size
cat("Looking for values more than cut off::::",hhat.cutoff,"\n")
for(i in c(1:n_sample_size)){
    if(hhat.model[i] > hhat.cutoff){
        cat("Validate these values for Influential points:::",hhat.model[i],"at observation",i,"\n")
    }
}
# we see only observation 831 as both outlier and influential point, so trying to remove it
data.train.treated <- data.train[-c(831),]
model3_transformed_treated <- lm(log(Item_Outlet_Sales)~Item_Fat_Content+Item_Type+Outlet_Identifier+Outlet_Establishment_Year+Outlet_Size+Outlet_Location_Type+Outlet_Type+Item_Weight+Item_Visibility+Item_MRP,data = data.train.treated)
summary(model3_transformed_treated)
# removing the outlier impoves the Adj R-square very significantly

# Ste-15: Model validation for Multicollinearity
# vif(model3_transformed) # No aliased coefficient in the model

# Step-16: Computing the standardized coefficient
#data.train.std <- sapply(data.train[,],FUN=scale)
#data.train.std <- data.frame(data.train)
#model3_transformed.std <- lm(log(Item_Outlet_Sales)~Item_Fat_Content+Item_Type+Outlet_Identifier+Outlet_Establishment_Year+Outlet_Size+Outlet_Location_Type+Outlet_Type+Item_Weight+Item_Visibility+Item_MRP, data = data.train)
#summary(model3_transformed.std)
#since most of the variables are factorial in nature, there is no need of standardizing the value

# Step-17: Model Validation
FINAL_MODEL <- lm(log(Item_Outlet_Sales) ~ Outlet_Identifier + Item_MRP, data = data.train)
final_summary <- summary(FINAL_MODEL); final_summary # adj r-square is 72.41%
str(data.test)
COUNT_PREDICTED <- predict(FINAL_MODEL,data.test)
plot(COUNT_PREDICTED,data.test$Item_Outlet_Sales,lwd=2, cex=2, col="red")
COUNT_PREDICTED_RE_TRANSFORMED <- exp(COUNT_PREDICTED)
plot(COUNT_PREDICTED_RE_TRANSFORMED,data.test$count,lwd=2, cex=2, col="green")
abline(0,1,col='red', lwd=2)

# Step-18: Prediction 
# Prediction Interval
pred_Int <- predict(FINAL_MODEL,data.test,interval = "predict")
conf_Int <- predict(FINAL_MODEL,data.test,interval = "confidence")
converted_pred_int <- exp(pred_Int)
converted_conf_int <- exp(conf_Int)
data.test$predicted_count <- converted_pred_int[,1]
data.test$prediction_interval_low <- converted_pred_int[,2]
data.test$prediction_interval_high <- converted_pred_int[,3]
data.test$confidence_interval_low <- converted_conf_int[,2]
data.test$confidence_interval_high <- converted_conf_int[,3]
data.prediction.result <- data.frame(data.test$Item_Outlet_Sales,data.test$predicted_count,data.test$prediction_interval_low,data.test$prediction_interval_high,data.test$confidence_interval_low,data.test$confidence_interval_high)
View(data.prediction.result)
data.test$predicted_count <- NULL
data.test$prediction_interval_low <- NULL
data.test$prediction_interval_high <- NULL
data.test$confidence_interval_low <- NULL
data.test$confidence_interval_high <- NULL

