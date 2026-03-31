# ============================================================
# Random Forest Classification in R: A Full Demo
# Example: iris data
# ============================================================

# This demo illustrates:
# 1. fitting a random forest classifier
# 2. examining out-of-bag (OOB) error
# 3. making test-set predictions
# 4. inspecting variable importance
# 5. visualizing how random forests improve on a single tree
#
# Package used: randomForest

# ------------------------------------------------------------
# 0. Install/load package
# ------------------------------------------------------------

# install.packages("randomForest")   # run once if needed
library(randomForest)

# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------

data(iris)

head(iris)
str(iris)
table(iris$Species)

# ------------------------------------------------------------
# 2. Train/test split
# ------------------------------------------------------------

set.seed(123)

n <- nrow(iris)
train_id <- sample(seq_len(n), size = round(0.7 * n))

iris_train <- iris[train_id, ]
iris_test  <- iris[-train_id, ]

# ------------------------------------------------------------
# 3. Fit a random forest classifier
# ------------------------------------------------------------

# Important arguments:
# - ntree: number of trees
# - mtry: number of predictors randomly considered at each split
# - importance = TRUE: compute variable importance measures

rf_fit <- randomForest(
  Species ~ .,
  data = iris_train,
  ntree = 500,
  mtry = 2,
  importance = TRUE
)

print(rf_fit)

# Interpretation:
# - OOB estimate of error rate is built into the fit
# - confusion matrix shown here is based on OOB predictions

# ------------------------------------------------------------
# 4. OOB error over trees
# ------------------------------------------------------------

plot(rf_fit, main = "Random forest OOB error vs number of trees")

# Teaching point:
# The OOB error usually stabilizes as the number of trees increases.
# This helps determine whether enough trees were used.

# ------------------------------------------------------------
# 5. OOB confusion matrix and OOB error
# ------------------------------------------------------------

cat("\nOOB confusion matrix:\n")
print(rf_fit$confusion)

cat("\nFinal OOB error rate:\n")
print(rf_fit$err.rate[rf_fit$ntree, "OOB"])

# ------------------------------------------------------------
# 6. Test-set predictions
# ------------------------------------------------------------

test_pred_class <- predict(rf_fit, newdata = iris_test, type = "class")
test_pred_prob  <- predict(rf_fit, newdata = iris_test, type = "prob")

cat("\nTest confusion matrix:\n")
print(table(True = iris_test$Species, Predicted = test_pred_class))

cat("\nTest accuracy:\n")
print(mean(test_pred_class == iris_test$Species))

cat("\nTest misclassification rate:\n")
print(mean(test_pred_class != iris_test$Species))

cat("\nFirst few test-set predicted probabilities:\n")
print(round(head(test_pred_prob), 3))

# ------------------------------------------------------------
# 7. Variable importance
# ------------------------------------------------------------

cat("\nVariable importance:\n")
print(importance(rf_fit))

# Two common importance measures:
# - MeanDecreaseAccuracy
# - MeanDecreaseGini

varImpPlot(rf_fit, main = "Variable Importance in Random Forest")

# Teaching point:
# Variables with larger importance values contribute more to prediction.
# On iris, petal variables are usually much more important.

# ------------------------------------------------------------
# 8. Compare with a smaller forest
# ------------------------------------------------------------

rf_small <- randomForest(
  Species ~ .,
  data = iris_train,
  ntree = 20,
  mtry = 2,
  importance = FALSE
)

cat("\nSmall forest OOB error rate:\n")
print(rf_small$err.rate[rf_small$ntree, "OOB"])

cat("\nLarge forest OOB error rate:\n")
print(rf_fit$err.rate[rf_fit$ntree, "OOB"])

# Teaching point:
# More trees usually improve stability, though after a point
# the gains become small.

# ------------------------------------------------------------
# 9. Effect of mtry
# ------------------------------------------------------------

rf_mtry1 <- randomForest(
  Species ~ .,
  data = iris_train,
  ntree = 500,
  mtry = 1
)

rf_mtry4 <- randomForest(
  Species ~ .,
  data = iris_train,
  ntree = 500,
  mtry = 4
)

cat("\nOOB error with mtry = 1:\n")
print(rf_mtry1$err.rate[rf_mtry1$ntree, "OOB"])

cat("\nOOB error with mtry = 2:\n")
print(rf_fit$err.rate[rf_fit$ntree, "OOB"])

cat("\nOOB error with mtry = 4:\n")
print(rf_mtry4$err.rate[rf_mtry4$ntree, "OOB"])

# Teaching point:
# mtry controls the amount of randomness at each split.
# Smaller mtry increases tree diversity.
# Larger mtry makes trees more similar to bagging.

# ------------------------------------------------------------
# 10. Compare random forest to a single classification tree
# ------------------------------------------------------------

# install.packages("rpart")      # run once if needed
# install.packages("rpart.plot") # run once if needed
library(rpart)

tree_fit <- rpart(
  Species ~ .,
  data = iris_train,
  method = "class"
)

tree_pred <- predict(tree_fit, newdata = iris_test, type = "class")

cat("\nSingle tree test accuracy:\n")
print(mean(tree_pred == iris_test$Species))

cat("\nRandom forest test accuracy:\n")
print(mean(test_pred_class == iris_test$Species))

# Teaching point:
# Random forests usually outperform a single tree because they
# reduce variance through averaging.

# ------------------------------------------------------------
# 11. Proximity matrix (optional advanced feature)
# ------------------------------------------------------------

rf_prox <- randomForest(
  Species ~ .,
  data = iris_train,
  ntree = 200,
  proximity = TRUE
)

# The proximity matrix measures how often two observations land
# in the same terminal node across trees.
prox_mat <- rf_prox$proximity

cat("\nDimensions of proximity matrix:\n")
print(dim(prox_mat))

# ------------------------------------------------------------
# 12. Multidimensional scaling plot from proximities
# ------------------------------------------------------------

mds <- cmdscale(1 - prox_mat, k = 2)

plot(mds,
     col = as.numeric(iris_train$Species),
     pch = 19,
     xlab = "Coordinate 1",
     ylab = "Coordinate 2",
     main = "MDS plot based on random forest proximities")

legend("topright",
       legend = levels(iris_train$Species),
       col = 1:3,
       pch = 19)

# Teaching point:
# This gives a visual sense of how the random forest sees
# similarity among observations.

# ------------------------------------------------------------
# 13. Predict new observations
# ------------------------------------------------------------

new_flowers <- data.frame(
  Sepal.Length = c(5.0, 6.5, 7.2),
  Sepal.Width  = c(3.4, 3.0, 3.2),
  Petal.Length = c(1.5, 4.8, 6.0),
  Petal.Width  = c(0.2, 1.8, 2.2)
)

new_pred_class <- predict(rf_fit, newdata = new_flowers, type = "class")
new_pred_prob  <- predict(rf_fit, newdata = new_flowers, type = "prob")

cat("\nPredicted classes for new flowers:\n")
print(new_pred_class)

cat("\nPredicted class probabilities for new flowers:\n")
print(round(new_pred_prob, 3))

# ------------------------------------------------------------
# 14. Notes for students
# ------------------------------------------------------------

# Key ideas:
# 1. A random forest is an ensemble of classification trees.
# 2. Each tree is fit to a bootstrap sample of the training data.
# 3. At each split, only a random subset of predictors is considered.
# 4. Predictions are made by majority vote across trees.
# 5. OOB error provides built-in validation.
# 6. Variable importance helps identify influential predictors.
#
# Why random forests often work well:
# - trees are flexible but high variance
# - averaging many randomized trees greatly improves stability