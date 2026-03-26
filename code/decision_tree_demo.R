# ============================================================
# Classification Tree in R using rpart
# Example: iris data
# ============================================================

# ------------------------------------------------------------
# 0. Install/load packages
# ------------------------------------------------------------

# install.packages("rpart")        # run once if needed
# install.packages("rpart.plot")   # run once if needed

library(rpart)
library(rpart.plot)

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
# 3. Fit a classification tree
# ------------------------------------------------------------

# method = "class" tells rpart this is a classification problem
tree_fit <- rpart(
  Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
  data = iris_train,
  method = "class"
)

# Basic printed summary
print(tree_fit)

# More detailed complexity table
printcp(tree_fit)

# Variable importance
tree_fit$variable.importance

# ------------------------------------------------------------
# 4. Plot the tree
# ------------------------------------------------------------

rpart.plot(
  tree_fit,
  type = 2,
  extra = 104,
  fallen.leaves = TRUE,
  main = "Classification Tree for iris"
)

# Notes:
# - Each internal node shows the splitting rule
# - Terminal nodes show predicted class and class probabilities
# - extra = 104 adds predicted class probabilities and percentages

# ------------------------------------------------------------
# 5. In-sample predictions
# ------------------------------------------------------------

train_pred_class <- predict(tree_fit, newdata = iris_train, type = "class")
train_pred_prob  <- predict(tree_fit, newdata = iris_train, type = "prob")

cat("\nTraining confusion matrix:\n")
print(table(True = iris_train$Species, Predicted = train_pred_class))

cat("\nTraining accuracy:\n")
print(mean(train_pred_class == iris_train$Species))

cat("\nFirst few training predicted probabilities:\n")
print(head(train_pred_prob))

# ------------------------------------------------------------
# 6. Test-set predictions
# ------------------------------------------------------------

test_pred_class <- predict(tree_fit, newdata = iris_test, type = "class")
test_pred_prob  <- predict(tree_fit, newdata = iris_test, type = "prob")

cat("\nTest confusion matrix:\n")
print(table(True = iris_test$Species, Predicted = test_pred_class))

cat("\nTest accuracy:\n")
print(mean(test_pred_class == iris_test$Species))

cat("\nTest misclassification rate:\n")
print(mean(test_pred_class != iris_test$Species))

# ------------------------------------------------------------
# 7. Inspect the complexity parameter table
# ------------------------------------------------------------

cat("\nComplexity parameter table:\n")
print(tree_fit$cptable)

# The cptable contains:
# - CP: complexity parameter
# - nsplit: number of splits
# - rel error: training error
# - xerror: cross-validated error
# - xstd: standard error of xerror

# A common approach:
# choose the CP corresponding to the smallest xerror
best_cp <- tree_fit$cptable[which.min(tree_fit$cptable[, "xerror"]), "CP"]

cat("\nBest CP based on minimum cross-validated error:\n")
print(best_cp)

# ------------------------------------------------------------
# 8. Prune the tree
# ------------------------------------------------------------

pruned_tree <- prune(tree_fit, cp = best_cp)

# Plot the pruned tree
rpart.plot(
  pruned_tree,
  type = 2,
  extra = 104,
  fallen.leaves = TRUE,
  main = "Pruned Classification Tree for iris"
)

# ------------------------------------------------------------
# 9. Evaluate the pruned tree
# ------------------------------------------------------------

pruned_pred_class <- predict(pruned_tree, newdata = iris_test, type = "class")
pruned_pred_prob  <- predict(pruned_tree, newdata = iris_test, type = "prob")

cat("\nPruned tree test confusion matrix:\n")
print(table(True = iris_test$Species, Predicted = pruned_pred_class))

cat("\nPruned tree test accuracy:\n")
print(mean(pruned_pred_class == iris_test$Species))

cat("\nPruned tree test misclassification rate:\n")
print(mean(pruned_pred_class != iris_test$Species))

# ------------------------------------------------------------
# 10. Compare original and pruned tree
# ------------------------------------------------------------

cat("\nOriginal tree number of terminal nodes:\n")
print(sum(tree_fit$frame$var == "<leaf>"))

cat("\nPruned tree number of terminal nodes:\n")
print(sum(pruned_tree$frame$var == "<leaf>"))

# ------------------------------------------------------------
# 11. Predicted probabilities for a few test observations
# ------------------------------------------------------------

cat("\nFirst 10 predicted probabilities from pruned tree:\n")
print(round(pruned_pred_prob[1:10, ], 3))

# ------------------------------------------------------------
# 12. Example predictions for new flowers
# ------------------------------------------------------------

new_flowers <- data.frame(
  Sepal.Length = c(5.0, 6.5, 7.2),
  Sepal.Width  = c(3.4, 3.0, 3.2),
  Petal.Length = c(1.5, 4.8, 6.0),
  Petal.Width  = c(0.2, 1.8, 2.2)
)

new_pred_class <- predict(pruned_tree, newdata = new_flowers, type = "class")
new_pred_prob  <- predict(pruned_tree, newdata = new_flowers, type = "prob")

cat("\nPredicted classes for new flowers:\n")
print(new_pred_class)

cat("\nPredicted class probabilities for new flowers:\n")
print(round(new_pred_prob, 3))

# ------------------------------------------------------------
# 13. Optional: simpler tree by controlling complexity up front
# ------------------------------------------------------------

small_tree <- rpart(
  Species ~ .,
  data = iris_train,
  method = "class",
  control = rpart.control(
    cp = 0.02,      # larger cp => simpler tree
    minsplit = 10,  # minimum observations before a split is attempted
    maxdepth = 3    # maximum tree depth
  )
)

rpart.plot(
  small_tree,
  type = 2,
  extra = 104,
  fallen.leaves = TRUE,
  main = "Smaller Classification Tree for iris"
)

small_pred <- predict(small_tree, newdata = iris_test, type = "class")

cat("\nSmaller tree test accuracy:\n")
print(mean(small_pred == iris_test$Species))

# ------------------------------------------------------------
# 14. Notes for students
# ------------------------------------------------------------

# Key ideas:
# 1. A classification tree recursively partitions the predictor space.
# 2. Each terminal node predicts the majority class in that region.
# 3. type = "class" gives class predictions.
# 4. type = "prob" gives estimated class probabilities.
# 5. Large trees can overfit, so pruning is important.
# 6. The cptable helps choose the level of pruning.