library(tidyverse)
creditcard <- read.csv("creditcard.csv")
creditcard %>%
  ggplot(aes(x = Class)) +
  geom_bar(color = "grey", fill = "lightgrey") +
  theme_bw()
summary(creditcard$Time)
# how many seconds are 24 hours
# 1 hr = 60 mins = 60 x 60 s = 3600 s
3600 * 24
# separate transactions by day
creditcard$day <- ifelse(creditcard$Time > 3600 * 24, "day2", "day1")

# make transaction relative to day
creditcard$Time_day <- ifelse(creditcard$day == "day2", creditcard$Time - 86400, creditcard$Time)

summary(creditcard[creditcard$day == "day1", ]$Time_day)
summary(creditcard[creditcard$day == "day2", ]$Time_day)
# bin transactions according to time of day
creditcard$Time <- as.factor(ifelse(creditcard$Time_day <= 38138, "gr1", # mean 1st Qu.
                                    ifelse(creditcard$Time_day <= 52327, "gr2", # mean mean
                                           ifelse(creditcard$Time_day <= 69580, "gr3", # mean 3rd Qu
                                                  "gr4"))))
creditcard %>%
  ggplot(aes(x = day)) +
  geom_bar(color = "grey", fill = "lightgrey") +
  theme_bw()

creditcard <- select(creditcard, -Time_day, -day)

# convert class variable to factor
creditcard$Class <- factor(creditcard$Class)
creditcard %>%
  ggplot(aes(x = Time)) +
  geom_bar(color = "grey", fill = "lightgrey") +
  theme_bw() +
  facet_wrap( ~ Class, scales = "free", ncol = 2)

summary(creditcard[creditcard$Class == "0", ]$Amount)
summary(creditcard[creditcard$Class == "1", ]$Amount)
creditcard %>%
  ggplot(aes(x = Amount)) +
  geom_histogram(color = "grey", fill = "lightgrey", bins = 50) +
  theme_bw() +
  facet_wrap( ~ Class, scales = "free", ncol = 2)

library(h2o)
h2o.init(nthreads = -1)
creditcard_hf <- as.h2o(creditcard)
splits <- h2o.splitFrame(creditcard_hf, 
                         ratios = c(0.4, 0.4), 
                         seed = 42)
train_unsupervised  <- splits[[1]]
train_supervised  <- splits[[2]]
test <- splits[[3]]

response <- "Class"
features <- setdiff(colnames(train_unsupervised), response)

model_nn <- h2o.deeplearning(x = features,
                             training_frame = train_unsupervised,
                             model_id = "model_nn",
                             autoencoder = TRUE,
                             reproducible = TRUE, #slow - turn off for real problems
                             ignore_const_cols = FALSE,
                             seed = 42,
                             hidden = c(10, 2, 10), 
                             epochs = 100,
                             activation = "Tanh")

h2o.saveModel(model_nn, path="model_nn", force = TRUE)
model_nn <- h2o.loadModel("model_nn")
model_nn
test_autoenc <- h2o.predict(model_nn, test)

train_features <- h2o.deepfeatures(model_nn, train_unsupervised, layer = 2) %>%
  as.data.frame() %>%
  mutate(Class = as.vector(train_unsupervised[, 31]))

ggplot(train_features, aes(x = DF.L2.C1, y = DF.L2.C2, color = Class)) +
  geom_point(alpha = 0.1)

# let's take the third hidden layer
train_features <- h2o.deepfeatures(model_nn, train_unsupervised, layer = 3) %>%
  as.data.frame() %>%
  mutate(Class = as.factor(as.vector(train_unsupervised[, 31]))) %>%
  as.h2o()
features_dim <- setdiff(colnames(train_features), response)

model_nn_dim <- h2o.deeplearning(y = response,
                                 x = features_dim,
                                 training_frame = train_features,
                                 reproducible = TRUE, #slow - turn off for real problems
                                 balance_classes = TRUE,
                                 ignore_const_cols = FALSE,
                                 seed = 42,
                                 hidden = c(10, 2, 10), 
                                 epochs = 100,
                                 activation = "Tanh")

h2o.saveModel(model_nn_dim, path="model_nn_dim", force = TRUE)
model_nn_dim <- h2o.loadModel("model_nn_dim/DeepLearning_model_R_1493574057843_49")
model_nn_dim
test_dim <- h2o.deepfeatures(model_nn, test, layer = 3)
h2o.predict(model_nn_dim, test_dim) %>%
  as.data.frame() %>%
  mutate(actual = as.vector(test[, 31])) %>%
  group_by(actual, predict) %>%
  summarise(n = n()) %>%
  mutate(freq = n / sum(n))
#Build neural networks with h20
#Trained an unsupervised neural network model using deep learning auto encoder.
#Bottleneck technique-hidden layer in middle is small-reduce the dimensionality of data(2 nodes/dimension)
#Get an accuracy of 83 percent. after dimensionality reduction