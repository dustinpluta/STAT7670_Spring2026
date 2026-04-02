# ============================================================
# XGBoost on Synthetic Binary Classification Data in R
# ============================================================
#
# Goal:
# Show how to use xgboost on a synthetic nonlinear classification problem.
#
# This demo illustrates:
# 1. generating synthetic data
# 2. fitting an xgboost classifier
# 3. evaluating test accuracy
# 4. plotting the decision boundary
# 5. inspecting variable importance
#
# ============================================================

# ------------------------------------------------------------
# 0. Install/load package
# ------------------------------------------------------------

# install.packages("xgboost")   # run once if needed
library(xgboost)

# ------------------------------------------------------------
# 1. Generate synthetic data
# ------------------------------------------------------------

set.seed(123)

generate_data <- function(n) {
  x1 <- runif(n, -3, 3)
  x2 <- runif(n, -3, 3)
  
  # noise predictors
  x3 <- rnorm(n)
  x4 <- rnorm(n)
  
  # nonlinear signal
  eta <- x1^2 + x2^2 + 1.5 * sin(2 * x1) - 4
  
  # convert to probability
  p <- 1 / (1 + exp(-eta))
  
  # generate binary response
  y <- rbinom(n, size = 1, prob = p)
  
  data.frame(x1 = x1, x2 = x2, x3 = x3, x4 = x4, y = y)
}

train_dat <- generate_data(500)
test_dat  <- generate_data(2000)

head(train_dat)
table(train_dat$y)

# ------------------------------------------------------------
# 2. Plot the training data
# ------------------------------------------------------------

plot(train_dat$x1, train_dat$x2,
     col = ifelse(train_dat$y == 1, "red", "black"),
     pch = 19,
     xlab = "x1",
     ylab = "x2",
     main = "Synthetic training data")

legend("topright",
       legend = c("Class 0", "Class 1"),
       col = c("black", "red"),
       pch = 19)

# ------------------------------------------------------------
# 3. Prepare xgboost input
# ------------------------------------------------------------

X_train <- as.matrix(train_dat[, c("x1", "x2", "x3", "x4")])
X_test  <- as.matrix(test_dat[, c("x1", "x2", "x3", "x4")])

y_train <- train_dat$y
y_test  <- test_dat$y

dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dtest  <- xgb.DMatrix(data = X_test, label = y_test)

# ------------------------------------------------------------
# 4. Fit xgboost model
# ------------------------------------------------------------

params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  max_depth = 3,
  eta = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8
)

xgb_fit <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 150,
  watchlist = list(train = dtrain, test = dtest),
  verbose = 0
)

print(xgb_fit)

# ------------------------------------------------------------
# 5. Predicted probabilities and classes
# ------------------------------------------------------------

test_prob <- predict(xgb_fit, newdata = dtest)
test_class <- ifelse(test_prob > 0.5, 1, 0)

# ------------------------------------------------------------
# 6. Evaluate performance
# ------------------------------------------------------------

cat("\nConfusion matrix:\n")
print(table(True = y_test, Predicted = test_class))

cat("\nTest accuracy:\n")
print(mean(test_class == y_test))

cat("\nTest misclassification rate:\n")
print(mean(test_class != y_test))

# ------------------------------------------------------------
# 7. Variable importance
# ------------------------------------------------------------

importance <- xgb.importance(
  feature_names = colnames(X_train),
  model = xgb_fit
)

cat("\nVariable importance:\n")
print(importance)

xgb.plot.importance(importance_matrix = importance)

# Teaching note:
# x1 and x2 should be most important, since x3 and x4 are noise variables.

# ------------------------------------------------------------
# 8. Plot fitted probability surface in x1-x2 plane
# ------------------------------------------------------------

x1_grid <- seq(-3, 3, length.out = 200)
x2_grid <- seq(-3, 3, length.out = 200)

grid <- expand.grid(x1 = x1_grid, x2 = x2_grid)

# Fix noise predictors at 0 for visualization
grid$x3 <- 0
grid$x4 <- 0

dgrid <- xgb.DMatrix(data = as.matrix(grid[, c("x1", "x2", "x3", "x4")]))

grid_prob <- predict(xgb_fit, newdata = dgrid)
z <- matrix(grid_prob, nrow = length(x1_grid), ncol = length(x2_grid))

# Plot observed data
plot(train_dat$x1, train_dat$x2,
     col = ifelse(train_dat$y == 1, "red", "black"),
     pch = 19,
     xlab = "x1",
     ylab = "x2",
     main = "XGBoost fitted probability boundary")

# Add probability contour at 0.5
contour(x1_grid, x2_grid, z,
        levels = 0.5,
        add = TRUE,
        drawlabels = FALSE,
        lwd = 2)

legend("topright",
       legend = c("Class 0", "Class 1"),
       col = c("black", "red"),
       pch = 19)

# ------------------------------------------------------------
# 9. Probability surface plot
# ------------------------------------------------------------

filled.contour(
  x1_grid, x2_grid, z,
  color.palette = terrain.colors,
  xlab = "x1",
  ylab = "x2",
  main = "Estimated P(Y = 1 | X) from XGBoost"
)

# ------------------------------------------------------------
# 10. Compare with a simple baseline: logistic regression
# ------------------------------------------------------------

glm_fit <- glm(y ~ x1 + x2 + x3 + x4,
               data = train_dat,
               family = binomial)

glm_prob <- predict(glm_fit, newdata = test_dat, type = "response")
glm_class <- ifelse(glm_prob > 0.5, 1, 0)

cat("\nLogistic regression test accuracy:\n")
print(mean(glm_class == y_test))

cat("\nXGBoost test accuracy:\n")
print(mean(test_class == y_test))

# Teaching note:
# Logistic regression is linear in the predictors,
# whereas xgboost can learn nonlinear interactions and threshold effects.

# ------------------------------------------------------------
# 11. Summary table
# ------------------------------------------------------------

comparison <- data.frame(
  Model = c("Logistic regression", "XGBoost"),
  Test_Accuracy = c(
    mean(glm_class == y_test),
    mean(test_class == y_test)
  ),
  Test_Error = c(
    mean(glm_class != y_test),
    mean(test_class != y_test)
  )
)

cat("\nComparison table:\n")
print(comparison)

# ------------------------------------------------------------
# 12. Predict new observations
# ------------------------------------------------------------

new_points <- data.frame(
  x1 = c(0, 2, -2),
  x2 = c(0, 1, -1),
  x3 = c(0, 0, 0),
  x4 = c(0, 0, 0)
)

dnew <- xgb.DMatrix(data = as.matrix(new_points))
new_prob <- predict(xgb_fit, newdata = dnew)
new_class <- ifelse(new_prob > 0.5, 1, 0)

cat("\nPredictions for new points:\n")
print(cbind(new_points,
            prob_class1 = round(new_prob, 3),
            pred_class = new_class))

# ------------------------------------------------------------
# 13. Notes for students
# ------------------------------------------------------------

# Key ideas:
# 1. xgboost fits trees sequentially, with each tree improving the fit.
# 2. It works especially well for nonlinear problems.
# 3. max_depth controls tree complexity.
# 4. eta controls the learning rate.
# 5. nrounds controls the number of boosting iterations.
# 6. Variable importance helps identify which predictors drive the model.