# Linear Discriminant Analysis (LDA) example in R
# Application setting:
# Classifying wine type from chemical measurements

# ----------------------------------
# 1. Load packages and data
# ----------------------------------
library(MASS)

data(iris)

# We'll use the built-in iris data.
# Predict species from the four flower measurements.

head(iris)
table(iris$Species)

# ----------------------------------
# 2. Quick exploratory plots
# ----------------------------------
pairs(iris[, 1:4], col = iris$Species, pch = 19)

# Group means by species
aggregate(. ~ Species, data = iris, mean)

# ----------------------------------
# 3. Fit the LDA model
# ----------------------------------
lda_fit <- lda(Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
               data = iris)

lda_fit

# Interpretation:
# - Prior probabilities
# - Group means
# - Coefficients of linear discriminants

# ----------------------------------
# 4. In-sample classification
# ----------------------------------
lda_pred <- predict(lda_fit)

names(lda_pred)
head(lda_pred$class)      # predicted class
head(lda_pred$posterior)  # posterior probabilities
head(lda_pred$x)          # discriminant scores

# Confusion matrix
table(True = iris$Species, Predicted = lda_pred$class)

# Apparent misclassification rate
mean(lda_pred$class != iris$Species)

# ----------------------------------
# 5. Plot discriminant scores
# ----------------------------------
plot(lda_pred$x,
     col = as.numeric(iris$Species),
     pch = 19,
     xlab = "LD1",
     ylab = "LD2",
     main = "LDA: Iris data")
legend("topright",
       legend = levels(iris$Species),
       col = 1:3,
       pch = 19)

# ----------------------------------
# 6. Train/test split example
# ----------------------------------
set.seed(123)

n <- nrow(iris)
train_id <- sample(1:n, size = round(0.7 * n))

iris_train <- iris[train_id, ]
iris_test  <- iris[-train_id, ]

lda_fit2 <- lda(Species ~ ., data = iris_train)

test_pred <- predict(lda_fit2, newdata = iris_test)

# Confusion matrix on test set
table(True = iris_test$Species, Predicted = test_pred$class)

# Test error rate
mean(test_pred$class != iris_test$Species)

# ----------------------------------
# 7. Compare to QDA
# ----------------------------------
qda_fit <- qda(Species ~ ., data = iris_train)
qda_pred <- predict(qda_fit, newdata = iris_test)

table(True = iris_test$Species, Predicted = qda_pred$class)
mean(qda_pred$class != iris_test$Species)

# ----------------------------------
# 8. Visualize only the first two variables
#    and overlay simple class regions
# ----------------------------------
lda_2var <- lda(Species ~ Petal.Length + Petal.Width, data = iris)

# Create grid
x1 <- seq(min(iris$Petal.Length) - 0.5, max(iris$Petal.Length) + 0.5, length.out = 200)
x2 <- seq(min(iris$Petal.Width)  - 0.5, max(iris$Petal.Width)  + 0.5, length.out = 200)
grid <- expand.grid(Petal.Length = x1, Petal.Width = x2)

grid_pred <- predict(lda_2var, newdata = grid)$class

# Convert factor classes to integers for plotting
z <- matrix(as.numeric(grid_pred), nrow = length(x1), ncol = length(x2))

plot(iris$Petal.Length, iris$Petal.Width,
     col = as.numeric(iris$Species),
     pch = 19,
     xlab = "Petal Length",
     ylab = "Petal Width",
     main = "LDA classification regions")

contour(x1, x2, z, add = TRUE, drawlabels = FALSE)

legend("topleft",
       legend = levels(iris$Species),
       col = 1:3,
       pch = 19)

# ----------------------------------
# 9. Inspect posterior probabilities
# ----------------------------------
head(test_pred$posterior)

# Look at uncertain cases
uncertainty <- apply(test_pred$posterior, 1, max)
head(sort(uncertainty))

# Cases with smallest maximum posterior probability
uncertain_cases <- iris_test[order(uncertainty), ]
head(uncertain_cases)

# ----------------------------------
# 10. Notes for students
# ----------------------------------
# LDA assumes:
# 1. Multivariate normality within each class
# 2. Equal covariance matrices across classes
# 3. Linear decision boundaries implied by those assumptions
#
# In practice:
# - lda_fit$scaling gives coefficients of the discriminant directions
# - predict(... ) returns class predictions, posterior probabilities,
#   and discriminant scores
# - LD1 and LD2 are the directions that best separate the classes