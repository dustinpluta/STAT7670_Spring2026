# -------------------------
# Setup: p = 3, n1 = n2 = 100
# -------------------------
set.seed(20260126)

p  <- 3
n1 <- 100
n2 <- 100

# Common covariance (positive definite)
Sigma <- matrix(c(
  1.0, 0.4, 0.2,
  0.4, 1.0, 0.3,
  0.2, 0.3, 1.0
), nrow = p, byrow = TRUE)

# Means: small shift to create marginal significance
mu1 <- c(0, 0, 0)
mu2 <- c(0.22, 0.18, 0.15)  # small multivariate difference

# Generate MVN samples (base-R via Cholesky)
rmvnorm_chol <- function(n, mu, Sigma) {
  L <- chol(Sigma)
  Z <- matrix(rnorm(n * length(mu)), nrow = n)
  sweep(Z %*% L, 2, mu, "+")
}

X <- rmvnorm_chol(n1, mu1, Sigma)
Y <- rmvnorm_chol(n2, mu2, Sigma)

# -------------------------
# Step 1: Sample means
# -------------------------
xbar <- colMeans(X)
ybar <- colMeans(Y)
d    <- xbar - ybar  # difference in sample means

cat("Step 1: sample means\n")
cat("xbar =", round(xbar, 4), "\n")
cat("ybar =", round(ybar, 4), "\n")
cat("d = xbar - ybar =", round(d, 4), "\n\n")

# -------------------------
# Step 2: Sample covariance matrices
# -------------------------
S1 <- cov(X)
S2 <- cov(Y)

cat("Step 2: sample covariance matrices\n")
cat("S1 =\n"); print(round(S1, 4))
cat("S2 =\n"); print(round(S2, 4))
cat("\n")

# -------------------------
# Step 3: Pooled covariance
# Sp = ((n1-1)S1 + (n2-1)S2) / (n1+n2-2)
# -------------------------
Sp <- ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)

cat("Step 3: pooled covariance Sp =\n")
print(round(Sp, 4))
cat("\n")

# -------------------------
# Step 4: Hotelling's T^2 statistic
# T2 = (n1*n2/(n1+n2)) * d' Sp^{-1} d
# -------------------------
invSp <- solve(Sp)

T2 <- (n1 * n2 / (n1 + n2)) * as.numeric(t(d) %*% invSp %*% d)

cat("Step 4: Hotelling's T^2\n")
cat("T2 =", round(T2, 6), "\n\n")

# -------------------------
# Step 5: Convert T^2 to F
# Under H0:
#   F = ((n1+n2-p-1) / (p*(n1+n2-2))) * T2
# with df1 = p and df2 = n1+n2-p-1
# -------------------------
df1 <- p
df2 <- n1 + n2 - p - 1

Fstat <- ((n1 + n2 - p - 1) / (p * (n1 + n2 - 2))) * T2

cat("Step 5: Convert to F\n")
cat("F =", round(Fstat, 6), "\n")
cat("df1 =", df1, ", df2 =", df2, "\n\n")

# -------------------------
# Step 6: p-value and decision at alpha=0.05
# -------------------------
pval <- 1 - pf(Fstat, df1, df2)

alpha <- 0.05
Fcrit <- qf(1 - alpha, df1, df2)

cat("Step 6: p-value and critical value\n")
cat("p-value =", signif(pval, 6), "\n")
cat("F critical (alpha=0.05) =", round(Fcrit, 6), "\n")

if (Fstat > Fcrit) {
  cat("Decision: Reject H0 at alpha=0.05\n")
} else {
  cat("Decision: Fail to reject H0 at alpha=0.05\n")
}
