library(MachineShop)
library(tidyverse)
library(recipes)
library(doParallel)
library(ggplot2); theme_set(theme_classic())

# Goals to predict
goal <- read_csv("subjectInfoChallenge.csv")
goal$`Subject-Id` <- as.character(goal$`Subject-Id`)
goal <- goal %>% 
  select(`Subject-Id`, Group, MADRS, YMRS) %>% 
  rename(record_id = `Subject-Id`)
goal$Group <- ifelse(goal$Group == "Bipolar", 1, 0)
goal$Group <- as.factor(goal$Group)
# Formatting training data
training <- read_csv("BipolarDerivedDataTraining.csv")
training <- training %>%
  select(record_id, starts_with('31P'))
training <- na.omit(training)
training$record_id <- as.character(training$record_id)
training <- training %>%
  select(-contains('gatpr')) %>%
  select(-contains('aatpr'))
training <- inner_join(goal, training)

goal <- training %>% select(record_id, Group, MADRS, YMRS)

## Extract meaningful column names
caudate <- colnames(training)[grep("caudacc", colnames(training))]
amygdala <- colnames(training)[grep("amygdala", colnames(training))]
hippocampus <- colnames(training)[grep("hippo", colnames(training))]
latofc <- colnames(training)[grep("latofc", colnames(training))]
mfg <- colnames(training)[grep("mfg", colnames(training))]
mpfc <- colnames(training)[grep("mfc", colnames(training))]
thalamus <- colnames(training)[grep("thal", colnames(training))]
cerebellum <- colnames(training)[grep("crbl", colnames(training))]
bstem <- colnames(training)[grep("brainstem", colnames(training))]

regions <- c(caudate, amygdala, latofc, mfg, mpfc, thalamus,
             cerebellum, cerebellum, bstem)
foo <- colnames(training)
rest <- setdiff(foo, regions)
rest <- rest[3:length(rest)]

# Recipes
base_rec <- recipe(Group ~ .,
                   data = training) %>%
  step_rm(record_id, MADRS, YMRS) %>%
  role_case(stratum = Group) %>%
  step_center(all_predictors()) %>%
  step_scale(all_predictors())

rec2 <- base_rec %>%
  step_spca(matches("caudacc"), num_comp = 2, prefix = "caudate", sparsity = 1) %>%
  step_spca(contains("amygdala"),num_comp = 2, prefix = "amygdala", sparsity = 1) %>%
  step_spca(contains("crbl"), num_comp = 2, prefix = "crbl", sparsity = 1) %>%
  step_spca(contains("hippo"), num_comp = 2, prefix = "hippo", sparsity = 1) %>%
  step_spca(contains("latofc"), num_comp = 2, prefix = "latofc", sparsity = 1) %>%
  step_spca(contains("mfg"), num_comp = 2, prefix = "mfg", sparsity = 1) %>%
  step_spca(contains("mfc"), num_comp = 2, prefix = "mpfc", sparsity = 1) %>%
  step_spca(contains("thal"), num_comp = 2, prefix = "thal", sparsity = 1) %>%
  step_spca(contains("brainstem"), num_comp = 2, prefix = "bstem", sparsity = 1) %>%
  step_spca(matches(rest), num_comp = 5, prefix = "everything_else", sparsity = 20) 

train_proc <- as.data.frame(juice(prep(rec2)))
train_proc$Group <- NULL
train_proc <- cbind(goal, train_proc)
# Formatting test data
testing <- read_csv("BipolarDerivedDataTesting.csv")
testing <- testing %>%
  select(record_id, starts_with('31P'))
testing <- na.omit(testing)
testing$record_id <- as.character(testing$record_id)
testing <- testing %>%
  select(-contains('gatpr')) %>%
  select(-contains('aatpr'))
testing$Group <- "Unknown"
testing$YMRS <- 0
testing$MADRS <- 0
testing$Group <- as.factor(testing$Group)

base_rec <- recipe(Group ~ .,
                   data = testing) %>%
  step_rm(record_id, MADRS, YMRS) %>%
  role_case(stratum = Group) %>%
  step_center(all_predictors()) %>%
  step_scale(all_predictors())

rec2 <- base_rec %>%
  step_spca(matches("caudacc"), num_comp = 2, prefix = "caudate", sparsity = 1) %>%
  step_spca(contains("amygdala"),num_comp = 2, prefix = "amygdala", sparsity = 1) %>%
  step_spca(contains("crbl"), num_comp = 2, prefix = "crbl", sparsity = 1) %>%
  step_spca(contains("hippo"), num_comp = 2, prefix = "hippo", sparsity = 1) %>%
  step_spca(contains("latofc"), num_comp = 2, prefix = "latofc", sparsity = 1) %>%
  step_spca(contains("mfg"), num_comp = 2, prefix = "mfg", sparsity = 1) %>%
  step_spca(contains("mfc"), num_comp = 2, prefix = "mpfc", sparsity = 1) %>%
  step_spca(contains("thal"), num_comp = 2, prefix = "thal", sparsity = 1) %>%
  step_spca(contains("brainstem"), num_comp = 2, prefix = "bstem", sparsity = 1) %>%
  step_spca(matches(rest), num_comp = 5, prefix = "everything_else", sparsity = 20)

test_proc <- as.data.frame(juice(prep(rec2)))
test_proc$Group <- "Unknown"
test_proc$YMRS <- 0
test_proc$MADRS <- 0
test_proc$Group <- as.factor(test_proc$Group)
record_id <- testing$record_id
test_proc <- cbind(test_proc, record_id)
test_proc$record_id <- as.character(test_proc$record_id)

###### Modeling 

status_rec <- recipe(Group ~ ., data = train_proc) %>%
  step_rm(record_id, MADRS, YMRS) %>%
  role_case(stratum = Group)

madrs_rec <- recipe(MADRS ~ ., data = train_proc) %>%
  step_rm(record_id, Group, YMRS) 

ymrs_rec <- recipe(YMRS ~ ., data = train_proc) %>%
  step_rm(record_id, MADRS, Group) 

# Status
grid <- 10
seed <- 3849382
control <- CVControl(folds = 5, seed = seed)

status_resamp <- resample(status_rec,
                          control = control,
                          metrics = auc,
                          model = RangerModel) 
status_perf <- performance(status_resamp, metrics = c("ROC AUC" = roc_auc,
                                                      "Sensitivity" = sensitivity,
                                                      "Specificity" = specificity))
plot(status_perf)
summary(performance(status_resamp))
status_mod_tuning <- TunedModel(RangerModel,
                     control = control,
                     grid = grid,
                     metrics = roc_auc)
status_mod_fit <- fit(status_rec,
                    model = status_mod_tuning)
status_imp <- varimp(status_mod_fit)
plot(status_imp)

# YMRS
grid <- 10
seed <- 3849382
control <- CVControl(folds = 5, seed = seed)

ymrs_resamp <- resample(ymrs_rec,
                          control = control,
                          metrics = rmse,
                          model = RangerModel) 
ymrs_perf <- performance(ymrs_resamp, metrics = c("RMSE" = rmse))
plot(ymrs_perf)
summary(performance(ymrs_resamp))
ymrs_mod_tuning <- TunedModel(RangerModel,
                                control = control,
                                grid = grid,
                                metrics = rmse)
ymrs_mod_fit <- fit(ymrs_rec,
                      model = ymrs_mod_tuning)
ymrs_imp <- varimp(ymrs_mod_fit)
plot(ymrs_imp)

# MADRS
grid <- 10
seed <- 3849382
control <- CVControl(folds = 5, seed = seed)

madrs_resamp <- resample(madrs_rec,
                        control = control,
                        metrics = rmse,
                        model = RangerModel) 
madrs_perf <- performance(madrs_resamp, metrics = c("RMSE" = rmse))
plot(madrs_perf)
summary(performance(madrs_resamp))
madrs_mod_tuning <- TunedModel(RangerModel,
                              control = control,
                              grid = grid,
                              metrics = rmse)
madrs_mod_fit <- fit(madrs_rec,
                    model = madrs_mod_tuning)
madrs_imp <- varimp(madrs_mod_fit)
plot(madrs_imp)

##### Predicting on test set
status_predict <- round(predict(status_mod_fit, newdata = test_proc, type  = "prob"), digits = 5)
ymrs_predict <- round(predict(ymrs_mod_fit, newdata = test_proc, type  = "prob"), digits = 5)
madrs_predict <- round(predict(madrs_mod_fit, newdata = test_proc, type  = "prob"), digits = 5)
predictions <- test_proc %>% select(record_id)
predictions <- cbind(predictions, status_predict, ymrs_predict, madrs_predict)
predictions <- predictions %>% arrange(record_id)
colnames(predictions) <- c("ID", "case_control", "YMRS", "MADRS")

# write.table(predictions, file = "predictions.txt",row.names = F, quote = F, sep = "\t")
predictions <- read.table(file = "predictions.txt", sep = "\t", header = T,
                          stringsAsFactors = F)
colnames(predictions) <- c("record_id", "case_control", "YMRS", "MADRS")
predictions$record_id <- as.character(predictions$record_id)

##### Comparing to true test labels
trues <- read.table("BipolarTestBlind.csv",
                  header = F, sep = ",")
colnames(trues) <- c("record_id", "case_control_true", "MADRS_true", "YMRS_true", "num_attempts")
trues$record_id <- as.character(trues$record_id)
trues$num_attempts <- NULL
colnames(predictions) <- c("record_id", "case_control_pred", "MADRS_pred", "YMRS_pred")

compare <- inner_join(predictions, trues)

