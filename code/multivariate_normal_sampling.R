library(MASS)

# Uncorrelated with equal variances

X <- mvrnorm(n = 100, c(1, 2), Sigma = matrix(c(1, 0, 0, 1), nrow = 2))
plot(X[, 1], X[, 2])

X <- mvrnorm(n = 500, c(1, 2), Sigma = matrix(c(1, 0, 0, 1), nrow = 2))
plot(X[, 1], X[, 2])

X <- mvrnorm(n = 1000, c(1, 2), Sigma = matrix(c(1, 0, 0, 1), nrow = 2))
plot(X[, 1], X[, 2])

X <- mvrnorm(n = 10000, c(1, 2), Sigma = matrix(c(1, 0, 0, 1), nrow = 2))
plot(X[, 1], X[, 2])

# Uncorrelated with different variances

X <- mvrnorm(n = 100, c(1, 2), Sigma = matrix(c(1, 0, 0, 4), nrow = 2))
plot(X[, 1], X[, 2])

X <- mvrnorm(n = 500, c(1, 2), Sigma = matrix(c(1, 0, 0, 4), nrow = 2))
plot(X[, 1], X[, 2])

X <- mvrnorm(n = 1000, c(1, 2), Sigma = matrix(c(1, 0, 0, 4), nrow = 2))
plot(X[, 1], X[, 2])

X <- mvrnorm(n = 10000, c(1, 2), Sigma = matrix(c(1, 0, 0, 4), nrow = 2))
plot(X[, 1], X[, 2])

# Correlated with unequal variances

X <- mvrnorm(n = 100, c(1, 2), Sigma = matrix(c(1, sqrt(2)/2, sqrt(2)/2, 1), nrow = 2))
plot(X[, 1], X[, 2])

X <- mvrnorm(n = 500, c(1, 2), Sigma = matrix(c(1, sqrt(2)/2, sqrt(2)/2, 1), nrow = 2))
plot(X[, 1], X[, 2])

X <- mvrnorm(n = 1000, c(1, 2), Sigma = matrix(c(1, sqrt(2)/2, sqrt(2)/2, 1), nrow = 2))
plot(X[, 1], X[, 2])

X <- mvrnorm(n = 10000, c(1, 2), Sigma = matrix(c(1, sqrt(2)/2, sqrt(2)/2, 1), nrow = 2))
plot(X[, 1], X[, 2])
