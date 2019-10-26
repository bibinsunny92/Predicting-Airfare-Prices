getwd();
setwd("C:/Users/Hp/Documents/R/Airfare")

#Installing necessary packages
install.packages("dplyr")
library(dplyr)
library(caTools)
library(ggplot2)
library(corrplot)
library(caret)
library(neuralnet)

# Loading the data and removing predictor variable
Airfare.data =  read.csv("Airfares.csv")
Airfare.data = Airfare.data[,-19]

# Getting the dimension ( no of rows and columns in our data)
dim(Airfare.data)  #638  19

# Structure of data
str(Airfare.data)

# Summary of data
summary(Airfare.data)

# Data Preprocessing
Airfare.data$FARE = as.numeric(substr(as.character(Airfare.data$FARE),2,10)) # To remove currency $

## S_INCOME AND E_INCOME are factors, we will need to convert it into number, and remove currency symbol

Airfare.data$S_INCOME = as.numeric(gsub("\\$|,","",Airfare.data$S_INCOME))
Airfare.data$E_INCOME = as.numeric(gsub("\\$|,","",Airfare.data$E_INCOME))

# Removing the unwanted columns that have text values
Airfare = Airfare.data[,5:18]                   

# corelation matrix
Airfare.corr = select_if(Airfare, is.numeric) # selecting the one which are numeric
coorelation = corrplot(cor(Airfare.corr), type = "upper", method = "number")

?plot
plot(x = Airfare$DISTANCE, y = Airfare$FARE,type = "p", main = "Relation between Distance and Fair", xlab = "Dist between 2 airports", ylab = "Avg Price")
plot(x = Airfare$COUPON, y = Airfare$FARE,type = "p", main = "Relation between No of Flights Stops and respective Fair", xlab = "No of Flights Stops", ylab = "Avg Price")

# converting dummy variables
DummyVar = dummyVars("~.",data = Airfare)
Airfare = data.frame(predict(DummyVar, newdata = Airfare))
str(Airfare)

## FEAUTURE SCALING

Airfare[,-18] <- lapply(Airfare[,-18], function(x) if(is.numeric(x)){(x - min(x))/(max(x) - min(x))} else x)

## Corelation matrix
Airfare.corr = select_if(Airfare, is.numeric) # selecting the one which are numeric
coorelation = corrplot(cor(Airfare), type = "upper", method = "number")

# SPLITTING DATA INTO TRAINING AND TESTING SETS
sampleSplit = sample.split(Airfare,SplitRatio = 0.8)
Airfare.training = subset(Airfare, sampleSplit == TRUE)
Airfare.test = subset(Airfare, sampleSplit == FALSE)


##### MODEL 1 - MULTIPLE LINEAR REGRESSION

set.seed(100)
modelLr = lm(FARE ~., data = Airfare.training)
summary(modelLr)

modelLR = lm(FARE ~ VACATION.No+SW.No+HI+E_INCOME+S_POP+E_POP+SLOT.Controlled+GATE.Constrained+DISTANCE+PAX, data = Airfare.training)
LR.predict = predict(modelLR, newdata = Airfare.test[,-18])

## Calculating the Accuracy
AccuracyLR = sum(abs(LR.predict - Airfare.test[,18]))/length(Airfare.test[,18])   #30.53392

## Plotting the graph
plot(modelLR)

##### MODEL 2 - DECISION TREE - REGRESSION MODEL
library(rpart)
set.seed(100)
modelDT = rpart(FARE ~.,data = Airfare.training)

varImp(modelDT, surrogates = FALSE, competes = TRUE)
DT.Predict = predict(modelDT, newdata = Airfare.test[,-18])

## Calculating the Accuracy
AccuracyDT = sum(abs(DT.Predict - Airfare.test[,18]))/length(Airfare.test[,18]) #27.08197

##### MODEL 3 - EXTREME GRADIENT BOOSTING
library(xgboost)

set.seed(100)

lab_matrix = as.matrix(Airfare.training$FARE)
data_trainX = as.matrix(Airfare.training[,-18])
dtrain = xgb.DMatrix(data = data_trainX, label = lab_matrix)
dim(as.matrix(Airfare.training$FARE))
dtest = xgb.DMatrix(data = as.matrix(Airfare.test[,-18]), label = as.matrix(Airfare.test$FARE))

# Defining the parameters
parameters = list(booster = "gblinear",
                  objective = "reg:linear",    
                  eta = 0.3,           #Vary btwn 0.1-0.3
                  nthread = 5,         #Increase this to improve speed
                  max_depth = 6,
                  lambda= 0.5,         #Vary between 0-3
                  alpha= 0.5,          #Vary between 0-3
                  min_child_weight= 2, #Vary btwn 1-10
                  eval_metric = "rmse")

watchlist = list(train = dtrain)

ntree = 150
set.seed(100)
model.Xgb = xgb.train(params = parameters, data = dtrain, watchlist = watchlist, nrounds = 53)

#model.Xgb = xgboost(params = parameters, data = dtrain, nrounds = 53)
model.Predict = predict(model.Xgb, dtest)

## Calculating the Accuracy
AccuracyXgb = sum(abs(model.Predict - Airfare.test[,18]))/length(Airfare.test[,18])  #34.28
AccuracyXgb

feautures = colnames(Airfare.training[-18])
impvar = xgb.importance(feautures, model = model.Xgb)


##### MODEL 4 - FITTING LASSO REGRESSION MODEL

install.packages("glmnet")
library(glmnet)
?glmnet
Airfare_X = as.matrix(Airfare[,-18])
Airfare_Y = Airfare$FARE
lasso_fit = glmnet(x = Airfare_X, y = Airfare_Y,family ="gaussian",alpha = 1 )

plot(lasso_fit, xvar = "lambda", label = TRUE)

cv_lasso = cv.glmnet(x = Airfare_X, y = Airfare_Y, family = "gaussian", alpha = 1, nfolds = 10)
plot(cv_lasso)

Lasso.Predict = predict(lasso_fit,newx = Airfare_X, s=cv_lasso$lambda.min)



Lasso.Predict1se = predict(lasso_fit, newx = Airfare_X, s=cv_lasso$lambda.1se)

## Calculating the Accuracy

AccuracyLS.min = sum(abs(Lasso.Predict - Airfare_Y))/length(Airfare_Y) #27.50271
AccuracyLS.1se = sum(abs(Lasso.Predict1se - Airfare_Y))/length(Airfare_Y) #28.71035

### Comparing the Accuracies to find best model

MACHINE_LEARNING_MODELS = c("Multiple Linear Regression","Decision Tree Model","Extreme Gradient Boosting","LASSO Regression")
ERROR = c(AccuracyLR,AccuracyDT,AccuracyXgb,AccuracyLS.min)

df = data.frame(MACHINE_LEARNING_MODELS,ERROR)
df




