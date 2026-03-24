# ============================================================
# Logistic Regression in R: A Full Demo
# ============================================================

# This demo uses the built-in iris data, converted to a
# binary classification problem:
#
#   Y = 1 if Species == "virginica"
#   Y = 0 otherwise
#
# The goal is to illustrate:
# 1. fitting a logistic regression model
# 2. interpreting coefficients
# 3. obtaining predicted probabilities
# 4. classifying observations
# 5. evaluating performance
# 6. visualizing the fitted probabilities

# ------------------------------------------------------------
# 1. Load and prepare the data
# ------------------------------------------------------------

data(iris)

iris_bin <- iris
iris_bin$IsVirginica <- ifelse(iris_bin$Species == "virginica", 1, 0)

head(iris_bin)
table(iris_bin$IsVirginica)

# ------------------------------------------------------------
# 2. Exploratory plots
# ------------------------------------------------------------

pairs(iris_bin[, 1:4],
      col = ifelse(iris_bin$IsVirginica == 1, "red", "black"),
      pch = 19,
      main = "Iris predictors: red = virginica, black = not virginica")

# ------------------------------------------------------------
# 3. Fit logistic regression
# ------------------------------------------------------------

# We model:
#   logit(P(Y=1 | X)) = beta0 + beta^T X

logit_fit <- glm(IsVirginica ~ Sepal.Length + Sepal.Width +
                   Petal.Length + Petal.Width,
                 data = iris_bin,
                 family = binomial)

summary(logit_fit)

# ------------------------------------------------------------
# 4. Interpreting coefficients
# ------------------------------------------------------------

coef(logit_fit)

# Odds ratios
exp(coef(logit_fit))

# Interpretation:
# A one-unit increase in predictor x_j changes the log-odds by beta_j
# and changes the odds by a factor exp(beta_j), holding the others fixed.

# ------------------------------------------------------------
# 5. Predicted probabilities
# ------------------------------------------------------------

pred_prob <- predict(logit_fit, type = "response")

head(pred_prob)

# These are estimated probabilities:
#   P(IsVirginica = 1 | X)

# ------------------------------------------------------------
# 6. Convert probabilities to class predictions
# ------------------------------------------------------------

pred_class <- ifelse(pred_prob > 0.5, 1, 0)

# Confusion matrix
cat("\nConfusion matrix:\n")
print(table(True = iris_bin$IsVirginica, Predicted = pred_class))

# Misclassification rate
cat("\nMisclassification rate:\n")
print(mean(pred_class != iris_bin$IsVirginica))

# Accuracy
cat("\nAccuracy:\n")
print(mean(pred_class == iris_bin$IsVirginica))

# ------------------------------------------------------------
# 7. Look at uncertain cases
# ------------------------------------------------------------

# Cases near 0.5 are the most uncertain
uncertainty <- abs(pred_prob - 0.5)
idx_uncertain <- order(uncertainty)[1:10]

cat("\nMost uncertain cases:\n")
print(iris_bin[idx_uncertain, c("Sepal.Length", "Sepal.Width",
                                "Petal.Length", "Petal.Width",
                                "Species", "IsVirginica")])
print(pred_prob[idx_uncertain])

# ------------------------------------------------------------
# 8. Train/test split
# ------------------------------------------------------------

set.seed(123)

n <- nrow(iris_bin)
train_id <- sample(seq_len(n), size = round(0.7 * n))

train_dat <- iris_bin[train_id, ]
test_dat  <- iris_bin[-train_id, ]

logit_train <- glm(IsVirginica ~ Sepal.Length + Sepal.Width +
                     Petal.Length + Petal.Width,
                   data = train_dat,
                   family = binomial)

test_prob <- predict(logit_train, newdata = test_dat, type = "response")
test_class <- ifelse(test_prob > 0.5, 1, 0)

cat("\nTest confusion matrix:\n")
print(table(True = test_dat$IsVirginica, Predicted = test_class))

cat("\nTest accuracy:\n")
print(mean(test_class == test_dat$IsVirginica))

cat("\nTest misclassification rate:\n")
print(mean(test_class != test_dat$IsVirginica))

# ------------------------------------------------------------
# 9. Visualize fitted probabilities using two predictors
# ------------------------------------------------------------

# For a simple visualization, fit a smaller model with two predictors
logit_2d <- glm(IsVirginica ~ Petal.Length + Petal.Width,
                data = iris_bin,
                family = binomial)

summary(logit_2d)

# Create a grid over predictor space
x1_seq <- seq(min(iris_bin$Petal.Length) - 0.5,
              max(iris_bin$Petal.Length) + 0.5,
              length.out = 200)

x2_seq <- seq(min(iris_bin$Petal.Width) - 0.5,
              max(iris_bin$Petal.Width) + 0.5,
              length.out = 200)

grid <- expand.grid(Petal.Length = x1_seq,
                    Petal.Width = x2_seq)

grid_prob <- predict(logit_2d, newdata = grid, type = "response")
z <- matrix(grid_prob, nrow = length(x1_seq), ncol = length(x2_seq))

# Plot observations
plot(iris_bin$Petal.Length, iris_bin$Petal.Width,
     col = ifelse(iris_bin$IsVirginica == 1, "red", "black"),
     pch = 19,
     xlab = "Petal Length",
     ylab = "Petal Width",
     main = "Logistic regression probabilities")

# Add contour lines for fitted probabilities
contour(x1_seq, x2_seq, z,
        levels = c(0.1, 0.25, 0.5, 0.75, 0.9),
        add = TRUE,
        drawlabels = TRUE)

legend("topleft",
       legend = c("Not virginica", "Virginica"),
       col = c("black", "red"),
       pch = 19)

# The 0.5 contour is the decision boundary

# ------------------------------------------------------------
# 10. ROC-style threshold exploration
# ------------------------------------------------------------

thresholds <- seq(0.1, 0.9, by = 0.1)

results <- data.frame(
  threshold = thresholds,
  sensitivity = NA,
  specificity = NA,
  accuracy = NA
)

for (i in seq_along(thresholds)) {
  th <- thresholds[i]
  pred_th <- ifelse(test_prob > th, 1, 0)
  
  TP <- sum(pred_th == 1 & test_dat$IsVirginica == 1)
  TN <- sum(pred_th == 0 & test_dat$IsVirginica == 0)
  FP <- sum(pred_th == 1 & test_dat$IsVirginica == 0)
  FN <- sum(pred_th == 0 & test_dat$IsVirginica == 1)
  
  results$sensitivity[i] <- TP / (TP + FN)
  results$specificity[i] <- TN / (TN + FP)
  results$accuracy[i] <- mean(pred_th == test_dat$IsVirginica)
}

cat("\nPerformance across thresholds:\n")
print(results)

# ------------------------------------------------------------
# 11. Example predictions for new flowers
# ------------------------------------------------------------

new_flowers <- data.frame(
  Sepal.Length = c(5.0, 6.5, 7.2),
  Sepal.Width  = c(3.4, 3.0, 3.2),
  Petal.Length = c(1.5, 4.8, 6.0),
  Petal.Width  = c(0.2, 1.8, 2.2)
)

new_prob <- predict(logit_fit, newdata = new_flowers, type = "response")
new_class <- ifelse(new_prob > 0.5, 1, 0)

cat("\nPredictions for new flowers:\n")
print(cbind(new_flowers,
            prob_virginica = round(new_prob, 4),
            pred_class = new_class))

# ------------------------------------------------------------
# 12. Notes for students
# ------------------------------------------------------------

# Logistic regression models:
#   P(Y=1 | X=x)
#
# through the logit link:
#   log(p / (1-p)) = beta0 + beta^T x
#
# Key outputs:
# - coefficients: effects on log-odds
# - exp(coefficients): odds ratios
# - predicted probabilities
# - classifications based on thresholding
#
# Important practical points:
# - the 0.5 threshold is conventional, not mandatory
# - logistic regression is primarily a probability model
# - classification comes from applying a cutoff to those probabilities