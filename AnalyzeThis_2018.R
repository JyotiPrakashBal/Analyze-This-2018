library(data.table)
library(dplyr)
library(lubridate)
library(mondate)
library(tidyverse)
library(polycor)
library(mice)
library(imputeTS)
library(rpart)
library(Amelia)
library(missForest)
library(doParallel)
library(caret)
library(xgboost)
library(h2o)
library(dplyr)
library(Rtsne)
library(dummies)
library(fastAdaboost)

credit <- read.csv("Training_dataset_Original.csv")
test_lead <- read.csv("Leaderboard_dataset.csv")
target <- credit[,49]

combine<- rbind(credit[,c(1:48)],test_lead)

#missing values

missing <- sapply(credit, function(x) sum(is.na(x)))

#converting missing values to NA

combine$mvar1[combine$mvar1=='na'] <- NA
combine$mvar2[combine$mvar2=='N/A'] <-NA
combine$mvar3[combine$mvar3=='N/A'] <- NA
combine$mvar4[combine$mvar4=='N/A'] <- NA
combine$mvar5[combine$mvar5=='N/A'] <- NA
combine$mvar6[combine$mvar6=='missing'] <- NA
combine$mvar7[combine$mvar7=='missing'] <- NA
combine$mvar8[combine$mvar8=='missing'] <- NA
combine$mvar9[combine$mvar9=='missing'] <- NA
combine$mvar10[combine$mvar10=='missing'] <- NA
combine$mvar11[combine$mvar11=='missing'] <- NA
combine$mvar12[combine$mvar12=='missing'] <- NA
combine$mvar13[combine$mvar13=='missing'] <- NA
combine$mvar14[combine$mvar14=='missing'] <- NA
combine$mvar15[combine$mvar15=='missing'] <- NA
combine$mvar16[combine$mvar16=='na'] <- NA
combine$mvar17[combine$mvar17=='na'] <- NA
combine$mvar18[combine$mvar18=='na'] <- NA
combine$mvar19[combine$mvar19=='na'] <- NA
combine$mvar20[combine$mvar20=='na'] <- NA
combine$mvar21[combine$mvar21=='N/A'] <- NA
combine$mvar22[combine$mvar22=='N/A'] <- NA
combine$mvar23[combine$mvar23=='N/A'] <- NA
combine$mvar24[combine$mvar24=='N/A'] <- NA
combine$mvar25[combine$mvar25=='missing'] <- NA
combine$mvar26[combine$mvar26=='missing'] <- NA
combine$mvar27[combine$mvar27=='missing'] <- NA
combine$mvar28[combine$mvar28=='missing'] <- NA
combine$mvar29[combine$mvar29=='missing'] <- NA
combine$mvar30[combine$mvar30=='missing'] <- NA
combine$mvar31[combine$mvar31=='missing'] <- NA
combine$mvar32[combine$mvar32=='missing'] <- NA
combine$mvar33[combine$mvar33=='N/A'] <- NA
combine$mvar34[combine$mvar34=='na'] <- NA
combine$mvar35[combine$mvar35=='na'] <- NA
combine$mvar36[combine$mvar36=='na'] <- NA
combine$mvar37[combine$mvar37=='na'] <- NA
combine$mvar38[combine$mvar38=='na'] <- NA
combine$mvar39[combine$mvar39=='na'] <- NA
combine$mvar40[combine$mvar40=='missing'] <- NA
combine$mvar41[combine$mvar41=='missing'] <- NA
combine$mvar42[combine$mvar42=='missing'] <- NA
combine$mvar43[combine$mvar43=='na'] <- NA
combine$mvar44[combine$mvar44=='N/A'] <- NA
combine$mvar45[combine$mvar45=='na'] <- NA
combine$mvar46[combine$mvar46=='na'] <- NA

missing <- sapply(credit, function(x) sum(is.na(x))/80000*100)
missing<- data.frame(missing)
#credit_mod <- credit

#categorical feature mvar47 one hot 

class(combine$mvar47)
combine$mvar47 <- as.numeric(factor(combine$mvar47))   # 'C' <- 1   'L' <- 2

#data conversion
combine[] = lapply(combine , as.character)
combine[] = lapply(combine , as.numeric)
#correlation <- cor(na.omit(combine))
credit$default_ind <- as.factor(credit$default_ind)
target<- as.factor(target)

#Missing value imputation

combine <- lapply(combine, function(x) ifelse(is.na(x), median(x, na.rm = TRUE), x))
combine <- data.frame(combine)

#credit$mvar2 <- ifelse(is.na(credit$mvar2), mean(credit$mvar2, na.rm=TRUE), credit$mvar2)

#combine <- read.csv("Iterative_Imputer_withoutlast.csv")

train <- combine[c(1:80000),]
test <- combine[c(80001:105000),]
train <- cbind(train,target)
train$mvar47 <- as.factor(train$mvar47)
test$mvar47 <- as.factor(test$mvar47)

combine <- combine %>% mutate_each_(funs(scale(.) %>% as.vector), 
                                vars=c(2:46))


###################################################

#Feature Engineering

new_my_data <- dummy.data.frame(combine[,c(2:47)])


pca.train <- new_my_data[1:nrow(combine),]
pca.test <- new_my_data[-(1:nrow(combine)),]

prin_comp <- prcomp(pca.train, scale. = T)
prin_comp$rotation[1:5,1:4]
std_dev <- prin_comp$sdev

#compute variance
pr_var <- std_dev^2

#check variance of first 10 components
pr_var[1:10]

prop_varex <- pr_var/sum(pr_var)
prop_varex[1:20]

plot(prop_varex, xlab = "Principal Component",
       ylab = "Proportion of Variance Explained",
       type = "b")

pca_comp <- data.frame(prin_comp$x)
pca_comp <- pca_comp[,1:2]

combine <- cbind(combine,pca_comp)

######################################################

#Clutering

k2 <- kmeans(combine[,c(2:5,42)], centers = 4, nstart = 25)

cluster_no <- data.frame(k2$cluster) 

k3 <- kmeans(combine[,c(14,15,33)], centers = 4, nstart = 25)

cluster_demog <- data.frame(k3$cluster) 


combine <- cbind(combine,cluster_no)
combine <- cbind(combine,cluster_demog)

#tsne <- Rtsne(combine[,c(10:55)], dims = 2, perplexity=30, verbose=TRUE, max_iter = 500,check_duplicates = FALSE)


##############################################

#manual features


combine <- combine %>% mutate_each_(funs(scale(.) %>% as.vector), 
                             vars=c("mvar1","mvar2","mvar3","mvar4","mvar5","mvar42"))

combine$score= dat$mvar1-dat$mvar2-dat$mvar3-dat$mvar4-dat$mvar5-dat$mvar42
combine$loans = combine$mvar45 + combine$mvar46
combine$percent_deli <- combine$mvar39/combine$mvar36
combine$uti_cc <- combine$mvar13/combine$mvar14
combine$less_uti_cc <- combine$mvar19- combine$mvar16
combine$less_uti_cl <- combine$mvar20- combine$mvar17
combine$due_per_month <- combine$mvar12/combine$mvar32
combine$debt_to_income <- combine$mvar11/combine$mvar14
combine$per_cl <- (combine$mvar36-combine$mvar38)/(combine$mvar36-combine$mvar38 + combine$mvar37 + combine$mvar34)


########################################

cl <- makeCluster(6)
registerDoParallel(cl)

set.seed(1)
gc()
trctrl = trainControl(method="repeatedcv",number=4,repeats=1,classProbs = TRUE,
                      summaryFunction = twoClassSummary)
#tuneGr= expand.grid(.eta=c(0.001),.max_depth=c(3,5),colsample_bytree=0.6,.nrounds=c(10000),.gamma=c(0),min_child_weight=1)

model_xgb_ent = train(default~.,
                        data=as.matrix(train[,-1]),
                        method="xgbLinear",
                        metric="ROC",
                        #tuneGrid=tuneGr,
                        trControl=trctrl)

stopCluster(cl)

output_xgb_ent=predict(model_xgb_ent,newdata=,type="prob")
#write.csv(output_xgb_ent19,"ent19.csv",row.names = FALSE)
=ifelse(output_xgb_ent19$Pos>0.9,1,0)

##############################

#best till now


combine$mvar47 <- as.factor(combine$mvar47)
combine$k2.cluster <- as.factor(combine$k2.cluster)
combine$k3.cluster <- as.factor(combine$k3.cluster)

train <- combine[c(1:80000),]
test <- combine[c(80001:105000),]

train <- cbind(train,target)
train$target <- as.factor(train$target)

train <- train[,c(2:48)]
test <- test[,c(2:47)]


localH2O <- h2o.init(nthreads = -1)
h2o.init()
train.h2o <- as.h2o(train) 
test.h2o <- as.h2o(test)

y.dep <- 47
x.indep <- c(2:46)



################################################################

#Evaluation Metric 

Metric_new <- function(pred , target1){
  target1$target <- as.character(target1$target)
  target1$target <- as.numeric(target1$target)
  actual = target1
  
  data_frame = cbind(pred , actual)
  data_frame$predict <- as.character(data_frame$predict)
  data_frame$predict <- as.numeric(data_frame$predict)
  
  score = 0
  price = 50000
  data_frame=data_frame[order(data_frame$p0,decreasing = TRUE),]
  
  for(i in 1:nrow(target1)){
    
    if(data_frame$predict[i]==data_frame$target[i] &&data_frame$predict[i] == 0 ){
      score = score +100 
      price  = price -5
      
    }
    else if(data_frame$predict[i]==data_frame$target[i] && data_frame$predict[i] == 1 ){
      price = price - 10
      score = score + 100
    }
    else{
      price = price -10
    }
    
    if(price <= 0){
      return(score)
    }
    
  }
  
  
  
  
  
}





########################################################################

#splitting data

combine <- data.frame(combine)

train <- combine[c(1:80000),]
test <- combine[c(80001:105000),]
train <- cbind(train,target)
train$mvar47 <- as.factor(train$mvar47)

set.seed(101) # Set Seed so that same sample can be reproduced in future also
# Now Selecting 75% of data as sample from total 'n' rows of the data  
library(caTools)
set.seed(101) 
sample = sample.split(train$mvar1, SplitRatio = 0.8)
training = subset(train, sample == TRUE)
validation  = subset(train, sample == FALSE)


##########################################################################


training$target<-as.factor(training$target)
validation$target <- as.factor(validation$target)
#train_ensemble$target <- as.factor(train_ensemble$target)
training$mvar47 <- as.factor(training$mvar47)
validation$mvar47 <-as.factor(validation$mvar47) 
#train_ensemble$mvar47 <- as.factor(train_ensemble$mvar47) 
localH2O <- h2o.init(nthreads = -1)
h2o.init()
train.h2o <- as.h2o(training) 
test.h2o <- as.h2o(validation[,2:48])
#train_ensemble.h2o <- as.h2o(train_ensemble[,2:48])
y.dep <- 49
x.indep <- c(2:48)









############################################################################

#GBM


  system.time(
    gbm.model <- h2o.gbm(y=y.dep, x=x.indep, training_frame = train.h2o, ntrees = 1700, max_depth = 6, learn_rate = 0.01, seed = 1122))
  
  predict.gbm <- as.data.frame(h2o.predict(gbm.model, test.h2o))
  
  test <- combine[c(80001:105000),]
  test = cbind(test,predict.gbm)
  #test %>% arrange(predict,desc(p0))
  
  test$prob <- test$p0 - test$p1
  o <- with(test, order(-prob))
  test <- test[o,]
  test <- test[,c(1,49)]
  write.csv(test,"Mission101_IITGuwahati_88.csv")
  



h2o.performance (gbm.model)
predict.gbm <- as.data.frame(h2o.predict(gbm.model, test.h2o))

#########################################################


train$target<-as.factor(train$target)

#train_ensemble$target <- as.factor(train_ensemble$target)
train$mvar47 <- as.factor(train$mvar47)
test$mvar47 <- as.factor(test$mvar47)

ntrees <- c(1500,1550,1600,1650,1700,1750,1800,1850)

train.h2o <- as.h2o(train)

test.h2o <- as.h2o(test[,-1])
y.dep <- 49
flag =105
x.indep <- c(2:48)
##


##0.02 tak ho gya hai bhai 


  
  for(i in ntrees){
    flag = flag+1
    system.time(
      gbm.model <- h2o.gbm(y=y.dep, x=x.indep, training_frame = train.h2o, ntrees = i, max_depth = 4, learn_rate = 0.01,seed = 1122)
    )
    predict.gbm <- as.data.frame(h2o.predict(gbm.model, test.h2o))
    pred= data.frame(predict.gbm)
    pred$ID <- test$application_key
    pred$factor <- pred$p0-pred$p1
    pred=pred[order(pred$factor,decreasing = TRUE),]
    pred = pred[,c(4,1)]
    str = sprintf("Mission101_IITGuwahati_%d.csv",flag)
    write.table(pred,str,sep = ',',row.names = FALSE,col.names = FALSE)
    print(i)
    #print(j)
  }
  






#####################################################

# Deeplearning hyperparamters

response <- "target"
predictors <- setdiff(names(train), response)


hyper_params <- list(
  activation=c("Rectifier","Tanh","Maxout","RectifierWithDropout","TanhWithDropout","MaxoutWithDropout"),
  hidden=list(c(20,20),c(50,50),c(30,30,30),c(25,25,25,25)),
  input_dropout_ratio=c(0,0.05),
  l1=seq(0,1e-4,1e-6),
  l2=seq(0,1e-4,1e-6)
)

search_criteria = list(strategy = "RandomDiscrete", max_runtime_secs = 120, max_models = 100, seed=1234567, stopping_rounds=5)


dl_random_grid <- h2o.grid(
  algorithm="deeplearning",
  grid_id = "dl_grid_random",
  training_frame=as.h2o(training),
  validation_frame=as.h2o(validation),
  x=predictors,
  y=response,
  epochs=1,
  stopping_metric="logloss",
  stopping_tolerance=1e-2, ## stop when logloss does not improve by >=1% for 2 scoring events
  stopping_rounds=2,
  score_validation_samples=10000, ## downsample validation set for faster scoring
  score_duty_cycle=0.025, ## don't score more than 2.5% of the wall time
  max_w2=10, ## can help improve stability for Rectifier
  hyper_params = hyper_params,
  search_criteria = search_criteria
)


grid <- h2o.getGrid("dl_grid_random")

#print(dl_gridperf)


# Note that that these results are not reproducible since we are not using a single core H2O cluster
# H2O's DL requires a single core to be used in order to get reproducible results

# Grab the model_id for the top DL model, chosen by validation AUC
max_score <- 0
flag=122
for(i in 1:29)
{
  best_dl_model_id <- grid@model_ids[[i]]
  best_dl <- h2o.getModel(best_dl_model_id)
  predict.dl <- as.data.frame(h2o.predict(best_dl, test.h2o))
  pred= data.frame(predict.dl)
  pred$ID <- test$application_key
  pred$factor <- pred$p0-pred$p1
  pred=pred[order(pred$factor,decreasing = TRUE),]
  pred = pred[,c(4,1)]
  str = sprintf("Mission101_IITGuwahati_%d.csv",flag)
  flag=flag+1
  write.table(pred,str,sep = ',',row.names = FALSE,col.names = FALSE)
  print(i)
  
}





predict.best_dl<- as.data.frame(h2o.predict(best_dl, test.h2o))

score = Metric_new(predict.best_dl,target)

#####################################

gbm_params2 <- list(learn_rate = seq(0.01, 0.1, 0.01),
                    max_depth = seq(2, 10, 1),
                    sample_rate = seq(0.5, 1.0, 0.1),
                    col_sample_rate = seq(0.1, 1.0, 0.1))
search_criteria2 <- list(strategy = "RandomDiscrete", 
                         max_models = 36)

# Train and validate a grid of GBMs
gbm_grid2 <- h2o.grid("gbm", x = predictors, y = response,
                      grid_id = "gbm_grid2",
                      training_frame = as.h2o(training),
                      validation_frame = as.h2o(validation),
                      ntrees = 100,
                      seed = 1,
                      hyper_params = gbm_params2,
                      search_criteria = search_criteria2)     

gbm_gridperf2 <- h2o.getGrid(grid_id = "gbm_grid2") 
 
print(gbm_gridperf2)



#########################################################################

#gbm with bayesian optimization

h2o_bayes <- function(
  max_depth, learn_rate, sample_rate, 
  col_sample_rate, balance_classes){
  bal.cl <- as.logical(balance_classes)
  gbm <- h2o.gbm(  
    x                   = predictors,
    y                   = response,
    training_frame      = as.h2o(training),
    validation_frame    = as.h2o(validation),
    #nfolds              = 3,
    ntrees              = 900,
    max_depth           = max_depth,
    learn_rate          = learn_rate,
    sample_rate         = sample_rate,
    col_sample_rate     = col_sample_rate,
    score_tree_interval = 5,
    stopping_rounds     = 2,
    stopping_metric     = "logloss",
    stopping_tolerance  = 0.005,
    balance_classes     = bal.cl)
  
  score <- h2o.auc(gbm, valid = T)
  list(Score = score,
       Pred  = 0)
}

library(rBayesianOptimization)

system.time(OPT_Res <- BayesianOptimization(
  h2o_bayes,
  bounds = list(
    max_depth   = c(2L, 8L), 
    learn_rate  = c(1e-4, 0.2),
    sample_rate = c(0.4, 1), 
    col_sample_rate = c(0.4, 1), 
    balance_classes = c(0L, 1L)),
  init_points = 10,  n_iter = 10,
  acq = "ucb", kappa = 2.576, eps = 0.0,
  verbose = FALSE))




gbm <- h2o.gbm(
  x                   = predictors,
  y                   = response,
  training_frame      = as.h2o(training),
  validation_frame    = as.h2o(validation),
  ntrees              = 900,
  max_depth           = 7,
  learn_rate          = 0.07040253,
  sample_rate         =  0.61906999,
  col_sample_rate     = 0.4,
  balance_classes     = as.logical(OPT_Res$Best_Par["balance_classes"]),
  score_tree_interval = 5,
  stopping_rounds     = 2,
  stopping_metric     = "logloss",
  stopping_tolerance  = 0.005,
  model_id         = "my_awesome_GBM")



predict.gbm <- as.data.frame(h2o.predict(gbm.model, test.h2o))






##########################################################################

test <- combine[c(80001:105000),]
test = cbind(test,predict.gbm)
#test %>% arrange(predict,desc(p0))

o <- with(test, order(predict,-p0))
test <- test[o,]

test <- test %>% mutate(rank1 = 1:n())

test$prob <- test$p0 - test$p1
o <- with(test, order(-prob))
test <- test[o,]
test <- test %>% mutate(rank2 = 1:n())


o <- with(test, order(p1))
test <- test[o,]
test <- test %>% mutate(rank3 = 1:n())

test <- test[,c(1,49)]
write.csv(test,"Mission101_IITGuwahati_87.csv")

##########################################

#stacked ensemble


row_count <- nrow(train)
shuffled_rows <- sample(row_count)
train_split <- train[head(shuffled_rows,floor(row_count*0.75)),]
valid_split <- train[tail(shuffled_rows,floor(row_count*0.25)),]




library(h2oEnsemble) 
y <- "target"
x <- setdiff(names(train), y)




##########################

train_new = train[,c(2:48)]
train$target <- factor(train$target)
train_new = data.frame(train_new)

system.time(test_adaboost <- adaboost(as.matrix(train_new), target, tree_depth = 3, n_rounds = 100, verbose = FALSE,control = NULL))

###################################

xgb_grid_1 = expand.grid(
  nrounds = 1000,
  eta = c(0.01, 0.001, 0.0001),
  max_depth = c(2, 4, 6, 8, 10),
  gamma = 1
)

# pack the training control parameters
xgb_trcontrol_1 = trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  returnData = FALSE,
  returnResamp = "all",                                                        # save losses across all models
  classProbs = TRUE,                                                           # set to TRUE for AUC to be computed
  summaryFunction = twoClassSummary,
  allowParallel = TRUE
)

# train the model for each parameter combination in the grid, 
#   using CV to evaluate
xgb_train_1 = train(
  x = as.matrix(train %>%
                  select(-application_key)),
  y = as.factor(train$target),
  trControl = xgb_trcontrol_1,
  tuneGrid = xgb_grid_1,
  method = "xgbTree"
)





#################################################
predict.en <- as.data.frame(h2o.predict(ensemble, test.h2o))
test <- combine[c(80001:105000),]
test = cbind(test,predict.en)
#test %>% arrange(predict,desc(p0))

o <- with(test, order(predict,-p0))
test <- test[o,]
write.csv(test,"Mission101_IITGuwahati_47.csv",row.names = FALSE)




