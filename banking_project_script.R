#banking_project

library(tidymodels)
library(visdat)
library(tidyr)
library(dplyr)
library(car)
library(pROC)
library(ggplot2)
library(vip)
library(rpart.plot)
library(DALEXtra)
library(recipes)
library(ranger)
library(randomForest)
bank_train=read.csv("C:/Users/HP/Downloads/FOR PROJECT SUBMISSION/banking_project/bank-full_train.csv",stringsAsFactors = FALSE)
bank_test=read.csv("C:/Users/HP/Downloads/FOR PROJECT SUBMISSION/banking_project/bank-full_test.csv",stringsAsFactors = FALSE)

bank_train$y=as.factor(bank_train$y)
glimpse(bank_train)

dp_pipe=recipe(y~.,data=bank_train) %>% 
  update_role(day,month,ID,new_role = "drop_vars") %>%
  update_role(job,marital,education,contact,poutcome,new_role="to_dummies") %>% 
  step_rm(has_role("drop_vars")) %>% 
  step_unknown(has_role("to_dummies"),new_level="__missing__") %>% 
  step_other(has_role("to_dummies"),threshold =0.02,other="__other__") %>% 
  step_dummy(has_role("to_dummies")) %>%
  step_impute_median(all_numeric(),-all_outcomes())

dp_pipe=prep(dp_pipe)

train=bake(dp_pipe,new_data=NULL)
test=bake(dp_pipe,new_data=bank_test)

#dtree

tree_model=decision_tree(
  cost_complexity = tune(),
  tree_depth = tune(),
  min_n = tune()
) %>%
  set_engine("rpart") %>%
  set_mode("classification")

folds = vfold_cv(train, v = 5)

tree_grid = grid_regular(cost_complexity(), tree_depth(),
                         min_n(), levels = 3)

# doParallel::registerDoParallel()
my_res=tune_grid(
  tree_model,
  y~.,
  resamples = folds,
  grid = tree_grid,
  metrics = metric_set(roc_auc),
  control = control_grid(verbose = TRUE)
)
#show_notes(.Last.tune.result)
autoplot(my_res)+theme_light()

fold_metrics=collect_metrics(my_res)

my_res %>% show_best()

final_tree_fit=tree_model %>% 
  finalize_model(select_best(my_res)) %>% 
  fit(y~.,data=train)

#feature importance
final_tree_fit %>%
  vip(geom = "col", aesthetics = list(fill = "midnightblue", alpha = 0.8)) +
  scale_y_continuous(expand = c(0, 0))

#plot the tree

rpart.plot(final_tree_fit$fit)

#predictions

train_pred=predict(final_tree_fit,new_data = train,type="prob") %>% select(.pred_yes)
test_pred=predict(final_tree_fit,new_data = test,type="prob") %>% select(.pred_yes)

#cutoff

train.score=train_pred$.pred_yes

real=train$y

# KS plot

rocit = ROCit::rocit(score = train.score, 
                     class = real) 

kplot=ROCit::ksplot(rocit,legend=F)

# cutoff on the basis of KS

my_cutoff=kplot$`KS Cutoff`

test_hard_class=as.numeric(test_pred>my_cutoff)

# submissin
library(MASS)

write.csv(test_hard_class,"C:/Users/HP/Downloads/FOR PROJECT SUBMISSION/banking_project/Vinay_Sharma_P5_part2.csv")

print(KS)

ks.test(test_pred,test, alternative='greater')$statistic

kplot("KS stat")

### random forest

## Random Forest

rf_model = rand_forest(
  mtry = tune(),
  trees = tune(),
  min_n = tune()
) %>%
  set_mode("classification") %>%
  set_engine("ranger")

folds = vfold_cv(train, v = 5)

rf_grid = grid_regular(mtry(c(5,25)), trees(c(100,500)),
                       min_n(c(2,10)),levels = 3)


my_res=tune_grid(
  rf_model,
  y~.,
  resamples = folds,
  grid = rf_grid,
  metrics = metric_set(roc_auc),
  control = control_grid(verbose = TRUE)
)


autoplot(my_res)+theme_light()

fold_metrics=collect_metrics(my_res)

my_res %>% show_best()

final_rf_fit=rf_model %>% 
  set_engine("ranger",importance='permutation') %>% 
  finalize_model(select_best(my_res,"roc_auc")) %>% 
  fit(Revenue.Grid~.,data=train)

# variable importance 

final_rf_fit %>%
  vip(geom = "col", aesthetics = list(fill = "midnightblue", alpha = 0.8)) +
  scale_y_continuous(expand = c(0, 0))

# predicitons

train_pred=predict(final_rf_fit,new_data = train,type="prob") %>% select(.pred_1)
test_pred=predict(final_rf_fit,new_data = test,type="prob") %>% select(.pred_1)

### finding cutoff for hard classes

train.score=train_pred$.pred_1

real=train$Revenue.Grid

# KS plot

rocit = ROCit::rocit(score = train.score, 
                     class = real) 

kplot=ROCit::ksplot(rocit)

# cutoff on the basis of KS

my_cutoff=kplot$`KS Cutoff`

## test hard classes 

test_hard_class=as.numeric(test_pred>my_cutoff)



