# ============================================================
# Support Vector Machine (SVM) in R: A Full Demo
# ============================================================

# This demo uses the built-in iris data, converted to a
# binary classification problem:
#
#   Y = "virginica" vs "not_virginica"
#
# The goal is to illustrate:
# 1. fitting a linear SVM
# 2. fitting a nonlinear SVM with an RBF kernel
# 3. obtaining class predictions
# 4. evaluating performance
# 5. visualizing the decision boundary in 2D
#
# We use the e1071 package.

# ------------------------------------------------------------
# 0. Install/load package
# ------------------------------------------------------------

# install.packages("e1071")   # run once if needed
library(e1071)

# ------------------------------------------------------------
# 1. Load and prepare the data
# ------------------------------------------------------------

data(iris)

iris_bin <- iris
iris_bin$Class <- ifelse(iris_bin$Species == "virginica",
                         "virginica", "not_virginica")
iris_bin$Class <- factor(iris_bin$Class)

table(iris_bin$Class)

# ------------------------------------------------------------
# 2. Exploratory look
# ------------------------------------------------------------

pairs(iris_bin[, 1:4],
      col = ifelse(iris_bin$Class == "virginica", "red", "black"),
      pch = 19,
      main = "Iris predictors: red = virginica, black = not virginica")

# ------------------------------------------------------------
# 3. Train/test split
# ------------------------------------------------------------

set.seed(123)

n <- nrow(iris_bin)
train_id <- sample(seq_len(n), size = round(0.7 * n))

train_dat <- iris_bin[train_id, ]
test_dat  <- iris_bin[-train_id, ]

# ------------------------------------------------------------
# 4. Fit a linear SVM
# ------------------------------------------------------------

svm_linear <- svm(Class ~ Sepal.Length + Sepal.Width +
                    Petal.Length + Petal.Width,
                  data = train_dat,
                  kernel = "linear",
                  cost = 1,
                  scale = TRUE)

print(svm_linear)

# ------------------------------------------------------------
# 5. Predict on the test set
# ------------------------------------------------------------

pred_linear <- predict(svm_linear, newdata = test_dat)

cat("\nLinear SVM confusion matrix:\n")
print(table(True = test_dat$Class, Predicted = pred_linear))

cat("\nLinear SVM accuracy:\n")
print(mean(pred_linear == test_dat$Class))

cat("\nLinear SVM misclassification rate:\n")
print(mean(pred_linear != test_dat$Class))

# ------------------------------------------------------------
# 6. Fit an RBF-kernel SVM
# ------------------------------------------------------------

svm_rbf <- svm(Class ~ Sepal.Length + Sepal.Width +
                 Petal.Length + Petal.Width,
               data = train_dat,
               kernel = "radial",
               cost = 1,
               gamma = 0.5,
               scale = TRUE)

print(svm_rbf)

pred_rbf <- predict(svm_rbf, newdata = test_dat)

cat("\nRBF SVM confusion matrix:\n")
print(table(True = test_dat$Class, Predicted = pred_rbf))

cat("\nRBF SVM accuracy:\n")
print(mean(pred_rbf == test_dat$Class))

cat("\nRBF SVM misclassification rate:\n")
print(mean(pred_rbf != test_dat$Class))

# ------------------------------------------------------------
# 7. Tune the SVM hyperparameters by cross-validation
# ------------------------------------------------------------

set.seed(123)

tuned_rbf <- tune(
  svm,
  Class ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
  data = train_dat,
  kernel = "radial",
  ranges = list(
    cost = c(0.1, 1, 10, 100),
    gamma = c(0.01, 0.1, 0.5, 1)
  )
)

cat("\nBest tuned model:\n")
print(tuned_rbf$best.model)

cat("\nBest tuning parameters:\n")
print(tuned_rbf$best.parameters)

pred_tuned <- predict(tuned_rbf$best.model, newdata = test_dat)

cat("\nTuned RBF SVM confusion matrix:\n")
print(table(True = test_dat$Class, Predicted = pred_tuned))

cat("\nTuned RBF SVM accuracy:\n")
print(mean(pred_tuned == test_dat$Class))

# ------------------------------------------------------------
# 8. Inspect support vectors
# ------------------------------------------------------------

cat("\nNumber of support vectors in linear SVM:\n")
print(nrow(svm_linear$SV))

cat("\nNumber of support vectors in tuned RBF SVM:\n")
print(nrow(tuned_rbf$best.model$SV))

# Teaching point:
# The fitted boundary is determined by the support vectors,
# not by all observations equally.

# ------------------------------------------------------------
# 9. Two-dimensional visualization
# ------------------------------------------------------------

# For visualization, use only two predictors:
# Petal.Length and Petal.Width
# These work very well for the iris classification problem.

train_2d <- train_dat[, c("Petal.Length", "Petal.Width", "Class")]
test_2d  <- test_dat[, c("Petal.Length", "Petal.Width", "Class")]

svm_2d <- svm(Class ~ Petal.Length + Petal.Width,
              data = train_2d,
              kernel = "radial",
              cost = 1,
              gamma = 0.5,
              scale = TRUE)

pred_2d <- predict(svm_2d, newdata = test_2d)

cat("\n2D RBF SVM accuracy:\n")
print(mean(pred_2d == test_2d$Class))

# ------------------------------------------------------------
# 10. Create a grid to visualize decision regions
# ------------------------------------------------------------

x1_seq <- seq(min(iris_bin$Petal.Length) - 0.5,
              max(iris_bin$Petal.Length) + 0.5,
              length.out = 200)

x2_seq <- seq(min(iris_bin$Petal.Width) - 0.5,
              max(iris_bin$Petal.Width) + 0.5,
              length.out = 200)

grid <- expand.grid(Petal.Length = x1_seq,
                    Petal.Width = x2_seq)

grid_pred <- predict(svm_2d, newdata = grid)
z <- matrix(as.numeric(grid_pred), nrow = length(x1_seq), ncol = length(x2_seq))

# Plot training data
plot(train_2d$Petal.Length, train_2d$Petal.Width,
     col = ifelse(train_2d$Class == "virginica", "red", "black"),
     pch = 19,
     xlab = "Petal Length",
     ylab = "Petal Width",
     main = "SVM decision regions (RBF kernel)")

# Add decision boundary contours
contour(x1_seq, x2_seq, z,
        levels = c(1.5),
        add = TRUE,
        drawlabels = FALSE,
        lwd = 2)

legend("topleft",
       legend = c("not_virginica", "virginica"),
       col = c("black", "red"),
       pch = 19)

# ------------------------------------------------------------
# 11. Highlight support vectors in the 2D plot
# ------------------------------------------------------------

points(svm_2d$SV[, "Petal.Length"],
       svm_2d$SV[, "Petal.Width"],
       pch = 1,
       cex = 1.8,
       lwd = 2)

# Support vectors are circled

# ------------------------------------------------------------
# 12. Compare linear vs radial in 2D
# ------------------------------------------------------------

svm_2d_linear <- svm(Class ~ Petal.Length + Petal.Width,
                     data = train_2d,
                     kernel = "linear",
                     cost = 1,
                     scale = TRUE)

pred_2d_linear <- predict(svm_2d_linear, newdata = test_2d)

cat("\n2D linear SVM accuracy:\n")
print(mean(pred_2d_linear == test_2d$Class))

cat("\n2D radial SVM accuracy:\n")
print(mean(pred_2d == test_2d$Class))

# ------------------------------------------------------------
# 13. Example predictions for new flowers
# ------------------------------------------------------------

new_flowers <- data.frame(
  Sepal.Length = c(5.0, 6.5, 7.2),
  Sepal.Width  = c(3.4, 3.0, 3.2),
  Petal.Length = c(1.5, 4.8, 6.0),
  Petal.Width  = c(0.2, 1.8, 2.2)
)

new_pred_linear <- predict(svm_linear, newdata = new_flowers)
new_pred_rbf    <- predict(tuned_rbf$best.model, newdata = new_flowers)

cat("\nPredictions for new flowers:\n")
print(cbind(new_flowers,
            linear_svm_pred = new_pred_linear,
            tuned_rbf_pred = new_pred_rbf))

# ------------------------------------------------------------
# 14. Notes for students
# ------------------------------------------------------------

# SVM constructs a decision boundary that maximizes margin.
#
# Key ideas:
# - kernel = "linear": linear separating hyperplane
# - kernel = "radial": nonlinear boundary
# - cost controls penalty for violations of the margin
# - gamma controls flexibility of the radial kernel
#
# Practical points:
# - tune() is commonly used to choose cost and gamma
# - support vectors are the observations that determine the boundary
# - SVM is a classifier, not inherently a probability model