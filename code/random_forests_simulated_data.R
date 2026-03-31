# ============================================================
# Random Forest vs Single Decision Tree on Synthetic Data
# ============================================================
#
# Goal of this demo:
# Show how a random forest can improve on a single decision tree
# by reducing variance and producing better generalization.
#
# We generate synthetic data with:
# - a nonlinear class boundary
# - some noise predictors
# - enough complexity that a single tree is unstable
#
# Packages used:
# - rpart
# - rpart.plot
# - randomForest
#
# ============================================================

# ------------------------------------------------------------
# 0. Install/load packages
# ------------------------------------------------------------

# install.packages("rpart")
# install.packages("rpart.plot")
# install.packages("randomForest")

library(rpart)
library(rpart.plot)
library(randomForest)

# ------------------------------------------------------------
# 1. Generate synthetic classification data
# ------------------------------------------------------------

set.seed(123)

n_train <- 500
n_test  <- 2000

generate_data <- function(n) {
  x1 <- runif(n, -3, 3)
  x2 <- runif(n, -3, 3)
  
  # Add noise predictors so that variable selection matters
  x3 <- rnorm(n)
  x4 <- rnorm(n)
  x5 <- runif(n, -2, 2)
  x6 <- rnorm(n)
  
  # Nonlinear signal:
  # class depends on a curved boundary in (x1, x2)
  eta <- x1^2 + x2^2 + 1.5 * sin(2 * x1) - 4
  
  # Convert to probability through logistic function
  p <- 1 / (1 + exp(-eta))
  
  # Sample class labels
  y <- rbinom(n, size = 1, prob = p)
  class <- factor(ifelse(y == 1, "Class1", "Class0"))
  
  data.frame(
    x1 = x1, x2 = x2, x3 = x3, x4 = x4, x5 = x5, x6 = x6,
    Class = class
  )
}

train_dat <- generate_data(n_train)
test_dat  <- generate_data(n_test)

head(train_dat)
table(train_dat$Class)
table(test_dat$Class)

# ------------------------------------------------------------
# 2. Plot the training data in the signal dimensions
# ------------------------------------------------------------

plot(train_dat$x1, train_dat$x2,
     col = ifelse(train_dat$Class == "Class1", "red", "black"),
     pch = 19,
     xlab = "x1",
     ylab = "x2",
     main = "Synthetic training data")

legend("topright",
       legend = c("Class0", "Class1"),
       col = c("black", "red"),
       pch = 19)

# Teaching note:
# The boundary is nonlinear, so a single shallow linear split
# structure will not be sufficient.

# ------------------------------------------------------------
# 3. Fit a single decision tree
# ------------------------------------------------------------

tree_fit <- rpart(
  Class ~ .,
  data = train_dat,
  method = "class",
  control = rpart.control(cp = 0.001, minsplit = 15)
)

print(tree_fit)
printcp(tree_fit)

rpart.plot(
  tree_fit,
  type = 2,
  extra = 104,
  fallen.leaves = TRUE,
  main = "Single Decision Tree"
)

# ------------------------------------------------------------
# 4. Evaluate the decision tree
# ------------------------------------------------------------

tree_pred_train <- predict(tree_fit, newdata = train_dat, type = "class")
tree_pred_test  <- predict(tree_fit, newdata = test_dat, type = "class")

tree_prob_test  <- predict(tree_fit, newdata = test_dat, type = "prob")

cat("\n=============================\n")
cat("Single Decision Tree Results\n")
cat("=============================\n")

cat("\nTraining confusion matrix:\n")
print(table(True = train_dat$Class, Predicted = tree_pred_train))

cat("\nTraining accuracy:\n")
print(mean(tree_pred_train == train_dat$Class))

cat("\nTest confusion matrix:\n")
print(table(True = test_dat$Class, Predicted = tree_pred_test))

cat("\nTest accuracy:\n")
print(mean(tree_pred_test == test_dat$Class))

cat("\nTest misclassification rate:\n")
print(mean(tree_pred_test != test_dat$Class))

# ------------------------------------------------------------
# 5. Fit a random forest
# ------------------------------------------------------------

set.seed(123)

rf_fit <- randomForest(
  Class ~ .,
  data = train_dat,
  ntree = 500,
  mtry = 2,
  importance = TRUE
)

print(rf_fit)

# ------------------------------------------------------------
# 6. Evaluate the random forest
# ------------------------------------------------------------

rf_pred_train <- predict(rf_fit, newdata = train_dat, type = "class")
rf_pred_test  <- predict(rf_fit, newdata = test_dat, type = "class")
rf_prob_test  <- predict(rf_fit, newdata = test_dat, type = "prob")

cat("\n=====================\n")
cat("Random Forest Results\n")
cat("=====================\n")

cat("\nTraining confusion matrix:\n")
print(table(True = train_dat$Class, Predicted = rf_pred_train))

cat("\nTraining accuracy:\n")
print(mean(rf_pred_train == train_dat$Class))

cat("\nOOB confusion matrix:\n")
print(rf_fit$confusion)

cat("\nFinal OOB error rate:\n")
print(rf_fit$err.rate[rf_fit$ntree, "OOB"])

cat("\nTest confusion matrix:\n")
print(table(True = test_dat$Class, Predicted = rf_pred_test))

cat("\nTest accuracy:\n")
print(mean(rf_pred_test == test_dat$Class))

cat("\nTest misclassification rate:\n")
print(mean(rf_pred_test != test_dat$Class))

# ------------------------------------------------------------
# 7. Compare tree vs random forest directly
# ------------------------------------------------------------

comparison <- data.frame(
  Model = c("Single tree", "Random forest"),
  Train_Accuracy = c(
    mean(tree_pred_train == train_dat$Class),
    mean(rf_pred_train == train_dat$Class)
  ),
  Test_Accuracy = c(
    mean(tree_pred_test == test_dat$Class),
    mean(rf_pred_test == test_dat$Class)
  ),
  Test_Error = c(
    mean(tree_pred_test != test_dat$Class),
    mean(rf_pred_test != test_dat$Class)
  )
)

cat("\n=====================\n")
cat("Model Comparison\n")
cat("=====================\n")
print(comparison)

# Teaching note:
# Usually the single tree will fit the training data fairly well
# but will have noticeably worse test performance than the random forest.
# The random forest improves generalization by averaging many trees.

# ------------------------------------------------------------
# 8. OOB error plot for the random forest
# ------------------------------------------------------------

plot(rf_fit, main = "Random Forest OOB Error")

# Teaching note:
# This shows how the OOB error stabilizes as the number of trees grows.

# ------------------------------------------------------------
# 9. Variable importance
# ------------------------------------------------------------

cat("\nVariable importance:\n")
print(importance(rf_fit))

varImpPlot(rf_fit, main = "Random Forest Variable Importance")

# Teaching note:
# x1 and x2 should emerge as the most important variables,
# while x3-x6 are mostly noise variables.

# ------------------------------------------------------------
# 10. Visualize decision boundaries in x1-x2 plane
# ------------------------------------------------------------
#
# We will fix the noise predictors at typical values and compare
# the classification regions from the single tree and the RF.

x1_grid <- seq(-3, 3, length.out = 250)
x2_grid <- seq(-3, 3, length.out = 250)

grid <- expand.grid(x1 = x1_grid, x2 = x2_grid)
grid$x3 <- 0
grid$x4 <- 0
grid$x5 <- 0
grid$x6 <- 0

grid_tree <- predict(tree_fit, newdata = grid, type = "class")
grid_rf   <- predict(rf_fit, newdata = grid, type = "class")

z_tree <- matrix(as.numeric(grid_tree), nrow = length(x1_grid), ncol = length(x2_grid))
z_rf   <- matrix(as.numeric(grid_rf),   nrow = length(x1_grid), ncol = length(x2_grid))

# Plot single tree regions
plot(train_dat$x1, train_dat$x2,
     col = ifelse(train_dat$Class == "Class1", "red", "black"),
     pch = 19,
     xlab = "x1",
     ylab = "x2",
     main = "Single Tree Decision Regions")

contour(x1_grid, x2_grid, z_tree,
        levels = 1.5,
        add = TRUE,
        drawlabels = FALSE,
        lwd = 2)

legend("topright",
       legend = c("Class0", "Class1"),
       col = c("black", "red"),
       pch = 19)

# Plot random forest regions
plot(train_dat$x1, train_dat$x2,
     col = ifelse(train_dat$Class == "Class1", "red", "black"),
     pch = 19,
     xlab = "x1",
     ylab = "x2",
     main = "Random Forest Decision Regions")

contour(x1_grid, x2_grid, z_rf,
        levels = 1.5,
        add = TRUE,
        drawlabels = FALSE,
        lwd = 2)

legend("topright",
       legend = c("Class0", "Class1"),
       col = c("black", "red"),
       pch = 19)

# Teaching note:
# The single tree gives blocky, axis-aligned regions.
# The random forest often produces a more flexible boundary by averaging
# many such trees.

# ------------------------------------------------------------
# 11. Probability surface comparison
# ------------------------------------------------------------

tree_prob_grid <- predict(tree_fit, newdata = grid, type = "prob")[, "Class1"]
rf_prob_grid   <- predict(rf_fit,   newdata = grid, type = "prob")[, "Class1"]

z_tree_prob <- matrix(tree_prob_grid, nrow = length(x1_grid), ncol = length(x2_grid))
z_rf_prob   <- matrix(rf_prob_grid,   nrow = length(x1_grid), ncol = length(x2_grid))

# Tree probability contours
filled.contour(
  x1_grid, x2_grid, z_tree_prob,
  color.palette = terrain.colors,
  xlab = "x1", ylab = "x2",
  main = "Single Tree Estimated P(Class1)"
)

# RF probability contours
filled.contour(
  x1_grid, x2_grid, z_rf_prob,
  color.palette = terrain.colors,
  xlab = "x1", ylab = "x2",
  main = "Random Forest Estimated P(Class1)"
)

# Teaching note:
# The tree gives coarse, stepwise probability regions.
# The random forest averages across trees and usually gives
# a smoother probability surface.

# ------------------------------------------------------------
# 12. Repeat over multiple simulations to show stability
# ------------------------------------------------------------

set.seed(999)

n_rep <- 30
acc_tree <- numeric(n_rep)
acc_rf   <- numeric(n_rep)

for (r in 1:n_rep) {
  tr <- generate_data(n_train)
  te <- generate_data(n_test)
  
  fit_tree <- rpart(
    Class ~ .,
    data = tr,
    method = "class",
    control = rpart.control(cp = 0.001, minsplit = 15)
  )
  
  fit_rf <- randomForest(
    Class ~ .,
    data = tr,
    ntree = 300,
    mtry = 2
  )
  
  pred_tree <- predict(fit_tree, newdata = te, type = "class")
  pred_rf   <- predict(fit_rf,   newdata = te, type = "class")
  
  acc_tree[r] <- mean(pred_tree == te$Class)
  acc_rf[r]   <- mean(pred_rf   == te$Class)
}

cat("\n========================================\n")
cat("Repeated Simulation Accuracy Comparison\n")
cat("========================================\n")

summary_df <- data.frame(
  Model = c("Single tree", "Random forest"),
  Mean_Test_Accuracy = c(mean(acc_tree), mean(acc_rf)),
  SD_Test_Accuracy   = c(sd(acc_tree), sd(acc_rf))
)

print(summary_df)

boxplot(acc_tree, acc_rf,
        names = c("Single tree", "Random forest"),
        main = "Test Accuracy Across Repeated Simulations",
        ylab = "Accuracy")

# Teaching note:
# This is often the most convincing part of the demo:
# random forests usually have both
# - higher average test accuracy
# - lower variability across repeated samples

# ------------------------------------------------------------
# 13. Final teaching summary
# ------------------------------------------------------------

cat("\n========================================\n")
cat("Teaching Summary\n")
cat("========================================\n")
cat("1. A single decision tree is flexible but unstable.\n")
cat("2. Random forests average many trees grown on bootstrap samples.\n")
cat("3. Random feature selection reduces correlation among trees.\n")
cat("4. The result is better generalization and more stable performance.\n")
cat("5. On this synthetic nonlinear problem, RF should outperform a single tree.\n")