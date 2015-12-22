
# Set working directory and load libraries --------------------------------------------------------------------------

setwd("D:/projects/telstra")
library(reshape2)
library(xgboost)

# Load data ---------------------------------------------------------------------------------------------------------

event_type <- read.csv("event_type.csv", header=T)
log_feature <- read.csv("log_feature.csv", header=T)
resource_type <- read.csv("resource_type.csv", header=T)
severity_type <- read.csv("severity_type.csv", header=T)
test <- read.csv("test.csv", header=T)
train <- read.csv("train.csv", header=T)

# Merge into wide-format train and test dataset --------------------------------------------------------------------

event_type$value <- 1
event_typeW <- dcast(event_type, id ~ event_type, value.var =  "value")
event_typeW[is.na(event_typeW)] <- 0

log_featureW <- dcast(log_feature, id ~ log_feature, value.var = "volume")
log_featureW[is.na(log_featureW)] <- 0

resource_type$value <- 1
resource_typeW <- dcast(resource_type, id ~ resource_type, value.var =  "value")
resource_typeW[is.na(resource_typeW)] <- 0

severity_type$value <- 1
severity_typeW <- dcast(severity_type, id ~ severity_type, value.var =  "value")
severity_typeW[is.na(severity_typeW)] <- 0

# Prepare train dataset

trainM <- merge(train, event_typeW, by="id")
trainM <- merge(trainM, log_featureW, by="id")
trainM <- merge(trainM, resource_typeW, by="id")
trainM <- merge(trainM, severity_typeW, by="id")

# Prepare test dataset

testM <- merge(test, event_typeW, by="id")
testM <- merge(testM, log_featureW, by="id")
testM <- merge(testM, resource_typeW, by="id")
testM <- merge(testM, severity_typeW, by="id")

# Model trainings --------------------------------------------------------------------------------------------------
# Xgboost fitting ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

xgb_train <- xgb.DMatrix(data=cbind(location = colsplit(trainM$location," ",
                                                        names=c("location_prefix","location_id"))[,2],
                           data.matrix(trainM[,-c(1,2,3)])),
                         label=trainM[,3])
                           
xgb_param <- list("objective" = "multi:softprob",
                  "eval_metric" = "mlogloss",
                  "num_class" = 3,
			            "max_depth" = 6,
			            "colsample_bytree" = 1,
			            "eta" = 0.1,
			            "min_child_weight" = 1)
        
xgb_cv <- xgb.cv(param=xgb_param, data=xgb_train_feature, label=xgb_train_response,
                 nfold=5, nrounds=500)

plot(xgb_cv$test.mlogloss.mean)
xgb_nround <- which(xgb_cv$test.mlogloss.mean==min(xgb_cv$test.mlogloss.mean))

# Xgboost fitting

xgb_fit <- xgb.train(data=xgb_train, param=xgb_param, nrounds=xgb_nround, verbose = 2)

# Xgboost predict

xgb_test <- cbind(location = colsplit(testM$location," ",names=c("location_prefix","location_id"))[,2],
                           data.matrix(testM[,-c(1,2)]))

xgb_predicted <- predict(xgb_fit, xgb_test)

xgb_pre_matrix <- data.frame(matrix(xgb_predicted, ncol=3,byrow=TRUE))
colnames(xgb_pre_matrix) <- paste0("predict_",0:2)

xgb_submission <- cbind(id = testM$id, xgb_pre_matrix[,c("predict_0","predict_1","predict_2")])

write.csv(xgb_submission, "submission.csv", quote=F, row.names=F)

# Logit fitting ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# formula <- fault_severity ~ . -id
# 
# fit <- glm(data=trainM,formula=formula,family=poisson)




# evaluation with multiclass logloss -------------------------------------------------------------------------------

LogLoss <- function(actual, predicted, eps=1e-15) {
        predicted[predicted < eps] <- eps;
        predicted[predicted > 1 - eps] <- 1 - eps;
        -1/nrow(actual)*(sum(actual*log(predicted)))
}


