######################################################################################
# Author: Pradeep Sathyamurthy
# Date: 07-June-2017
# Course: CSC-433
# Guiding Prof: Prof. Steve Jost
# Project: Final Project Submission
# Train Dataset Name: mart_train.csv
# Test Dataset Name: mart_test.csv
######################################################################################

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
plot(data.frame(table(data.mart.raw$Item_Fat_Content)), main="Frequency Distribution of Item_Fat_Content")
data.frame(table(data.mart.raw$Item_Type))
plot(data.frame(table(data.mart.raw$Item_Type)), main="Frequency Distribution of Item_Type")
data.frame(table(data.mart.raw$Outlet_Identifier))
plot(data.frame(table(data.mart.raw$Outlet_Identifier)), main="Frequency Distribution of Outlet_Identifier")
data.frame(table(data.mart.raw$Outlet_Size))
plot(data.frame(table(data.mart.raw$Outlet_Size)), main="Frequency Distribution of Outlet_Size")
data.frame(table(data.mart.raw$Outlet_Location_Type))
plot(data.frame(table(data.mart.raw$Outlet_Location_Type)), main="Frequency Distribution of Outlet_Location_Type")
data.frame(table(data.mart.raw$Outlet_Type))
plot(data.frame(table(data.mart.raw$Outlet_Type)), main="Frequency Distribution of Outlet_Type")
data.frame(table(data.mart.raw$Outlet_Establishment_Year))
plot(data.frame(table(data.mart.raw$Outlet_Establishment_Year)), main="Frequency Distribution of Outlet_Establishment_Year")


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


