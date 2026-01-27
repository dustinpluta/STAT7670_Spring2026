set.seed(20260126)

# -----------------------------
# Problem setup
# -----------------------------
p <- 3
n <- 100
alpha <- 0.05

# Choose a linear functional l' mu
l <- c(1, -0.5, 0.25)  # example l in R^3

# Generate fake MVN data: X_i ~ N_p(mu, Sigma)
mu_true <- c(0.5, -0.2, 0.1)
Sigma <- matrix(c(
  1.0, 0.3, 0.2,
  0.3, 1.0, 0.4,
  0.2, 0.4, 1.0
), nrow = p, byrow = TRUE)

# Base-R MVN sampler via Cholesky
rmvnorm_chol <- function(n, mu, Sigma) {
  L <- chol(Sigma)                 # Sigma = t(L) %*% L
  Z <- matrix(rnorm(n * length(mu)), nrow = n)
  sweep(Z %*% L, 2, mu, "+")
}

X <- rmvnorm_chol(n, mu_true, Sigma)

# -----------------------------
# Step 1: Compute xbar and S
# -----------------------------
xbar <- colMeans(X)     # sample mean (p-vector)
S <- cov(X)             # sample covariance (p x p)

cat("Step 1: xbar =\n"); print(round(xbar, 4))
cat("Step 1: S =\n");   print(round(S, 4))
cat("\n")

# -----------------------------
# Step 2: Point estimate for l' mu is l' xbar
# -----------------------------
theta_hat <- as.numeric(t(l) %*% xbar)
cat("Step 2: theta_hat = l' xbar =", round(theta_hat, 6), "\n\n")

# -----------------------------
# Step 3: Compute l' S l
# -----------------------------
lSl <- as.numeric(t(l) %*% S %*% l)
cat("Step 3: l' S l =", round(lSl, 6), "\n\n")

# -----------------------------
# Step 4: Get the F critical value for Hotelling ellipsoid
# -----------------------------
Fcrit <- qf(1 - alpha, df1 = p, df2 = n - p)
cat("Step 4: F_{p,n-p}(1-alpha) =", round(Fcrit, 6), "\n\n")

# -----------------------------
# Step 5: Compute the simultaneous CI half-width
#
# halfwidth = sqrt( [p(n-1)/(n(n-p))] * Fcrit * (l' S l) )
# -----------------------------
mult <- (p * (n - 1)) / (n * (n - p))
halfwidth <- sqrt(mult * Fcrit * lSl)

cat("Step 5: multiplier p(n-1)/(n(n-p)) =", round(mult, 6), "\n")
cat("Step 5: halfwidth =", round(halfwidth, 6), "\n\n")

# -----------------------------
# Step 6: Simultaneous confidence interval for l' mu
# -----------------------------
CI <- c(theta_hat - halfwidth, theta_hat + halfwidth)
names(CI) <- c("lower", "upper")

cat("Step 6: (1-alpha) simultaneous CI for l' mu:\n")
print(round(CI, 6))

# Optional: show the true l' mu for this simulated example
theta_true <- as.numeric(t(l) %*% mu_true)
cat("\nTrue l' mu (simulation truth) =", round(theta_true, 6), "\n")
