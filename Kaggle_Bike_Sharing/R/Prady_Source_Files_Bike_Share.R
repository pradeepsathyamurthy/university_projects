# Analyzing the dataset
getwd()
setwd("D:/Courses/CSC423 - SAS & R - Data Analysis and Regression/SAS/Project")
list.files(getwd())

# Importing the dataset
data.kaggle <- read.csv("train.csv",na.strings="Not Available",stringsAsFactors=FALSE)
head(data.kaggle,3)

# Variable type identification and finding missing values
str(data.kaggle)
table(is.na(data.kaggle))

# EDA on dataset
par(mfrow=c(4,2))
par(mar = rep(2, 4))
hist(data.kaggle$season)
hist(data.kaggle$weather)
hist(data.kaggle$humidity)
hist(data.kaggle$holiday)
hist(data.kaggle$workingday)
hist(data.kaggle$temp)
hist(data.kaggle$atemp)
hist(data.kaggle$windspeed)
prop.table(table(data.kaggle$weather))

########################################
# Data Splitting (Train:Test = 80:20)
########################################

# Installing the necessary library package
require(caret)
# Creating a random index to split the data as 80 - 20%
#idx <- createDataPartition(data.kaggle$count, p=.80, list=FALSE)
#print(idx)
# Using the index created to create a Training Data set - 131 observations created
#data.train <- data.kaggle[idx,]
#head(data.train)
# Using the index created to create a Testing Data set - 31 observations created
#data.test <- data.kaggle[-idx,]
#head(data.test)
#idx <- NULL

#############################################
# Dummy Variable or Factor variable creation
#############################################

table(data.train$season) # has 4 levels, thus need of 3 dummy variables
table(data.train$holiday) # this is just a binary variable
table(data.train$workingday) # this is again a binary variable
table(data.train$weather) # has 4 levels, thus need of 3 dummy variables

str(data.train)
data.train$season <- as.factor(data.train$season)
data.train$holiday <- as.factor(data.train$holiday)
data.train$workingday <- as.factor(data.train$workingday)
data.train$weather <- as.factor(data.train$weather)

###########################################################################
# Model Building - Ignoring datetime variable and using all other variable
###########################################################################

model1_count <- lm(count~season+holiday+workingday+weather+temp+atemp+humidity+windspeed+casual+registered, data=data.train)
summary(model1_count) # This gives Adj-R2 = 1, this is because variable casual + registred can calculate the count perfectly
# So features casual and registered are part of dependent variable and should be excluded from model
model2_count <- lm(count~season+holiday+workingday+weather+temp+atemp+humidity+windspeed, data=data.train)
summary(model2_count)
# model equation
# count = 115.3123 - 39.2224*(season3) + 64.9691*(season4) + 13.5484*(weather2) + 7.7457*(temp) + 3.0836*(atemp) -2.7415*(humidity) + 0.6616*(windspeed)
# However, adj R^2 is 0.2716

plot(model2_count)
df.temp <- data.frame(data.train$temp,data.train$atemp,data.train$humidity,data.train$windspeed)
cor(df.temp)
#                        data.train.temp data.train.atemp data.train.humidity data.train.windspeed
#data.train.temp           1.00000000       0.98519193         -0.06857852          -0.02005605
#data.train.atemp          0.98519193       1.00000000         -0.04880235          -0.05791124
#data.train.humidity      -0.06857852      -0.04880235          1.00000000          -0.31812732
#data.train.windspeed     -0.02005605      -0.05791124         -0.31812732           1.00000000
# we clearly see a strong corelation between temp and atemp
# building a model by removing atemp
require(car)
vif(model2_count)
model3_count <- lm(count~season+holiday+workingday+weather+temp+humidity+windspeed, data=data.train)
summary(model3_count)
# Adj-R^2 is still 0.2711

# we will try to deploy even variable selction method and try to build a optimum model
model3_count_stepwise <- step(model3_count, direction = "both")
summary(model3_count_stepwise)
# count = 120.2626 - 41.5517*(season3) + 65.3086*(season4) + 13.3975*(weather2) - 13.3334*(weather3) + 11.1348*(temp) - 2.7235*(humidity) + 0.5443*(windspeed)
# However, adj R^2 is 0.2712

#########################################################################################
# Bringing datetime variable into consideration for model building - Feature Engineering
#########################################################################################

str(data.train)
# we see datatime feature is of type "chr" with sample value like "2011-01-01 01:00:00"
# let us try to split this into Years, Month, weekdays and hours

# Subsetting hours from a datetime field
data.train$hour <- substr(data.train$datetime,12,13)
data.train$hour<- as.factor(data.train$hour)

# Subsetting weekday
date <- substr(data.train$datetime,1,10)
# Creating days from date
data.train$day <- weekdays(as.Date(date))
data.train$day <- as.factor(data.train$day)

# Seperating Month from the date
data.train$month=substr(data.train$datetime,6,7)
data.train$month=as.factor(data.train$month)

# Seperating Years from the date
data.train$year=substr(data.train$datetime,1,4)
data.train$year=as.factor(data.train$year)


##################################################################################################
# Model Building - Including variables created from datetime that is hour, weekday, Month and Year
##################################################################################################

model4_count <- lm(count~season+holiday+workingday+weather+temp+humidity+windspeed+hour+day+year+month, data=data.train)
#summary(model4_count)
model4_count_stepwise <- step(model4_count,direction = "both")
summary(model4_count_stepwise)
# This increases adj R^2 to 0.6911

model4_count_backward <- step(model4_count,direction = "backward")
summary(model4_count_backward)

model5 <- lm(count~weather+temp+humidity+windspeed+hour+day+year+month, data = data.train)
summary(model5)
sqrt(anova(model5))
# Adj R^2 is 0.6911

# Plotting model
par(mfrow=c(4,2))
par(mar = rep(2, 4))
plot(data.train$temp,data.train$count)
plot(data.train$humidity,data.train$count)
plot(data.train$windspeed,data.train$count)

# Interaction Model
model5_interaction <- lm(count~weather+temp*humidity*windspeed+hour+day+year+month, data = data.train)
summary(model5_interaction)
sqrt(anova(model5_interaction))

# Polynomial Model
model5_polynomial <- lm(count~weather+temp*temp+humidity*humidity+windspeed*windspeed+hour+day+year+month, data = data.train)
summary(model5)
sqrt(anova(model5_polynomial))

#########################################################################################
# Residual Analysis
#########################################################################################
par(mfrow=c(4,2))
par(mar = rep(2, 4))
plot(model5)
sd(data.train$count)
residual <- rstandard(model5)
hist(residual)
# Ensures presence of heteroscadasity, so lets try with transformation

#########################################################################################
# Log Transformation
#########################################################################################
# Doing log transformation on dependent variable
model6 <- lm(log(count)~weather+temp+humidity+windspeed+hour+day+year+month, data = data.train)
summary(model6)
# Adj R^2 is 0.8284
# Amazing, great improvemnt
par(mfrow=c(4,2))
par(mar = rep(2, 4))
plot(model6)

# Trying with few interaction terms
model7 <- lm(log(count)~weather+temp*humidity*windspeed+hour+day+year+month, data = data.train)
summary(model7)
# Adj R^2 is 0.8288, this is not a significant improvement over the ordinary linear model, so we can drop it

#########################################################################################
# OUtlier and Influential Point Check
#########################################################################################

# computing studentized residual for outlier check
require(MASS)
n_sample_size <- nrow(data.train)
studentized.residuals <- studres(model6)
#cat("Complete list of Studentized Residual::::","\n")
#print(studentized.residuals)
for(i in c(1:n_sample_size)){
    if(studentized.residuals[i] < -3 || studentized.residuals[i] > 3){
        cat("Validate these values for outliers:::",studentized.residuals[i],"at observation",i,"\n")
    }
}

# Influential Points
hhat.model <- lm.influence(model6)$hat
n_sample_size <- nrow(data.train)
p_beta <- length(model6$coefficients) +1
#cat("Complete list of HHat Values::::","\n")
#print(hhat.model)
hhat.cutoff <- (2*p_beta)/n_sample_size
cat("Looking for values more than cut off::::",hhat.cutoff,"\n")
for(i in c(1:n_sample_size)){
    if(hhat.model[i] > hhat.cutoff){
        cat("Validate these values for Influential points:::",hhat.model[i],"at observation",i,"\n")
    }
}
# None of the outliers are part of influential points, this reconfirms these are natural outliers
# Hence, we cannot remove the same

#########################################################################################
# Reconfirming the absence of multicollinearity
#########################################################################################
# Checking Variance Inflation Factor
require(car)
vif(model6)

#########################################################################################
# Computing the standardized coefficients
#########################################################################################
data.train.std <- sapply(data.train[,],FUN=scale)
data.train.std <- data.frame(data.train)
model6.final.std <- lm(log(count)~weather+temp*humidity*windspeed+hour+day+year+month, data = data.train)
summary(model6.final.std)


####################
#Model Validation
####################

FINAL_MODEL <- model6
final_summary <- summary(FINAL_MODEL); final_summary

# Feature engineering on Test Data
data.test$season <- as.factor(data.test$season)
data.test$holiday <- as.factor(data.test$holiday)
data.test$workingday <- as.factor(data.test$workingday)
data.test$weather <- as.factor(data.test$weather)
data.test$hour <- substr(data.test$datetime,12,13)
data.test$hour<- as.factor(data.test$hour)
date <- substr(data.test$datetime,1,10)
data.test$day <- weekdays(as.Date(date))
data.test$day <- as.factor(data.test$day)
data.test$month=substr(data.test$datetime,6,7)
data.test$month=as.factor(data.test$month)
data.test$year=substr(data.test$datetime,1,4)
data.test$year=as.factor(data.test$year)
str(data.test)

#Prediction
COUNT_PREDICTED <- predict(FINAL_MODEL,data.test)
plot(COUNT_PREDICTED,data.test$count,lwd=2, cex=2, col="red")
COUNT_PREDICTED_RE_TRANSFORMED <- exp(COUNT_PREDICTED)
plot(COUNT_PREDICTED_RE_TRANSFORMED,data.test$count,lwd=2, cex=2, col="green")
abline(0,1,col='red', lwd=2)

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
data.prediction.result <- data.frame(data.test$count,data.test$predicted_count,data.test$prediction_interval_low,data.test$prediction_interval_high,data.test$confidence_interval_low,data.test$confidence_interval_high)
View(data.prediction.result)
data.test$predicted_count <- NULL
data.test$prediction_interval_low <- NULL
data.test$prediction_interval_high <- NULL
data.test$confidence_interval_low <- NULL
data.test$confidence_interval_high <- NULL