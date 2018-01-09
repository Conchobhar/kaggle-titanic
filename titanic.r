library("data.table")
library("Metrics")
library("ROCR")
library("gbm")

################################################################################
# Open data
################################################################################
setwd("~/R/titanic/")
file_train  <- "~/R/titanic/train.csv"
file_test   <- "~/R/titanic/test.csv"
file_gender <- "~/R/titanic/gender_submission.csv"

data_full <- data.table(read.csv(file_train))

################################################################################
# Data Changes & Feature Creation
################################################################################
#   Replace NA Age values with median value from entire dataset
data_full[is.na(Age), Age := median(data_full[,Age], na.rm = TRUE)]


# Create age catagories - Young/ Old + Child/ Adult
data_full$Age_group[data_full$Age <= 6 ] <- "child_young"
data_full$Age_group[data_full$Age > 6  & data_full$Age <= 15] <- "child_old"
data_full$Age_group[data_full$Age > 15 & data_full$Age <= 30] <- "adult_young"
data_full$Age_group[data_full$Age > 30 & data_full$Age <= 50] <- "adult_old"
data_full$Age_group[data_full$Age > 50] <- "old"
data_full$Age_group <- as.factor(data_full$Age_group)


# Split data into random train and holdout samples
set.seed(142857)
trainRows <- sample( 1:nrow(data_full), floor( nrow(data_full) * 0.8))

data_train <- data_full[trainRows]
data_hold  <- data_full[-trainRows]

################################################################################
# Exploration
################################################################################
summary(data_train)
sort(unique(data_train[,Age]))

# Histograms
hist(data_train[,Age], breaks=25)
hist(data_train[which(data_train[,Age] < 25),Age])

plot(data_train[,Pclass],data_train[,Age])
data_train[which(data_train[,Age] == 20),Pclass]

# Age vs Pclass
hist(data_train[which(data_train[,Pclass] == 1),Age],breaks=20)
hist(data_train[which(data_train[,Pclass] == 2),Age],breaks=20)
hist(data_train[which(data_train[,Pclass] == 3),Age],breaks=20)


################################################################################
# GLM function
################################################################################
myFormula <- as.formula( Survived 
    ~ Pclass + Sex + poly(Age, degree = 3) + SibSp + Parch  + Fare 
)

GLM   <- glm(myFormula, data = data_train)

# People are more likely to have not survived. Guess no one survived as a 
# baseline prediction accuracy benchmark.
pred_baseline <- rep(0,length(data_train$Survived))
pred_train  <- predict(GLM,  newdata = data_train) <= 0.5
pred_hold   <- predict(GLM,  newdata = data_hold)  <= 0.5

err_train = sum(pred_train == data_train[,Survived])/length(pred_train)
err_hold  = sum(pred_hold == data_hold[,Survived])/length(pred_hold)

# Not applicable!
tmse <- mse(data_train[,Survived], pred_train)
hmse <- mse(data_hold[,Survived],  pred_hold)

################################################################################
# GBM
################################################################################
myFormula <- as.formula( Survived 
    ~ Pclass + Sex + Age  + SibSp + Parch  + Fare 
     + Cabin + Embarked
)


tick <- Sys.time()
myGBM <- gbm(myFormula, data = data_train,
                
                distribution      = "bernoulli",
                n.trees           = 1000,
                shrinkage         = 0.01,
                
                interaction.depth = 3,
                bag.fraction      = 0.5,
                train.fraction    = 0.8,
                
                n.minobsinnode    = 10,
                #cv.folds         = 5,
                keep.data         = TRUE,
                verbose           = TRUE,
                n.cores           = 1
)
Sys.time() - tick

summary(myGBM)

# Try and range of cutoff values and look at the resulting error rate
temp <- c()
for( cutoff in seq(0.1,0.9,0.1)){
  pred_train  <- as.integer((predict(myGBM,  newdata = data_train) >= cutoff))
  pred <- prediction(pred_train, data_train$Survived)
  error <- performance(pred,"err")
  temp  <- rbind(temp,c(error@y.values[[1]][2],cutoff))  
}


pred_train  <- as.integer(predict(myGBM,  newdata = data_train) >= 0.5)
pred_hold   <- as.integer(predict(myGBM,  newdata = data_hold)  >= 0.5)

################################################################################
# Precision and Recall via ROCR
################################################################################
pred <- prediction(pred_train, data_train$Survived)
# Recall-Precision curve             
RP.perf <- performance(pred, "prec", "rec");
plot (RP.perf);

# ROC curve
ROC.perf <- performance(pred, "tpr", "fpr");
plot (ROC.perf);

# ROC area under the curve
auc.tmp <- performance(pred,"auc");
auc <- as.numeric(auc.tmp@y.values)

#
err_train = sum(pred_train == data_train$Survived)/length(pred_train)
err_hold  = sum(pred_hold  == data_hold$Survived) /length(pred_hold)

sum(pred_train == data_train[,Survived])/length(pred_train)*100
sum(pred_hold == data_hold[,Survived])/length(pred_hold)*100





# Results scrapbook:

# without age group
# > sum(pred_train == data_train[,Survived])/length(pred_train)*100
# [1] 82.02247
# > sum(pred_hold == data_hold[,Survived])/length(pred_hold)*100
# [1] 78.77095
# 
# # with age group
# > sum(pred_train == data_train[,Survived])/length(pred_train)*100
# [1] 80.05618
# > sum(pred_hold == data_hold[,Survived])/length(pred_hold)*100
# [1] 78.77095