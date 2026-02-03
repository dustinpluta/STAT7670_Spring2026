# Multivariate-response linear regression in R
# Example: two continuous outcomes (y1, y2) regressed on predictors x1, x2
# Shows: fitting, coefficient extraction, residual covariance, joint (MANOVA-style) tests,
# and some basic diagnostics.

set.seed(123)

# -----------------------------
# 1) Simulate correlated outcomes
# -----------------------------
n <- 150

x1 <- rnorm(n)
x2 <- rbinom(n, size = 1, prob = 0.5)     # binary predictor
X  <- cbind(1, x1, x2)

# True coefficient matrix B: (intercept, x1, x2) x (y1, y2)
B <- matrix(c(
  1.0,  0.5,   # intercepts for y1,y2
  2.0, -1.0,   # x1 effects
  0.8,  0.2    # x2 effects
), nrow = 3, byrow = TRUE)

# Correlated errors: Sigma (2x2)
Sigma <- matrix(c(1.0, 0.6,
                  0.6, 1.5), nrow = 2, byrow = TRUE)

# Generate E ~ N(0, Sigma) row-wise
# Use Cholesky: if Z ~ N(0, I), then Z %*% chol(Sigma) has cov Sigma
Z <- matrix(rnorm(n * 2), n, 2)
E <- Z %*% chol(Sigma)

# Generate responses Y = X B + E
Y <- X %*% B + E
colnames(Y) <- c("y1", "y2")

dat <- data.frame(y1 = Y[,1], y2 = Y[,2], x1 = x1, x2 = factor(x2))

# -----------------------------
# 2) Fit multivariate response regression
# -----------------------------
# In base R, fit with lm() using cbind(y1, y2) on the LHS
fit <- lm(cbind(y1, y2) ~ x1 + x2, data = dat)

# Inspect fitted object
summary(fit)          # gives separate univariate summaries for each response
coef(fit)             # coefficient matrix: predictors x responses

# -----------------------------
# 3) Residual covariance estimate (Sigma_hat)
# -----------------------------
# Residuals are n x q matrix
R <- residuals(fit)   # matrix with columns y1,y2
q <- ncol(R)
p <- length(coef(fit)[,1])  # number of regression parameters (incl intercept)

# SSE/SSP (E matrix in MANOVA notation): E = R'R
E_hat <- t(R) %*% R

# Sigma_hat = (1 / (n - p)) * R'R
Sigma_hat <- E_hat / (n - p)

Sigma_hat

# -----------------------------
# 4) Joint hypothesis tests (MANOVA-style)
# -----------------------------
# This tests predictors jointly across the multivariate response vector.
# You can request Wilks, Pillai, Hotelling-Lawley, Roy.
man <- manova(cbind(y1, y2) ~ x1 + x2, data = dat)

summary(man, test = "Wilks")
summary(man, test = "Pillai")
summary(man, test = "Hotelling-Lawley")
summary(man, test = "Roy")

# If you want a single-term multivariate test, use update() or anova()
# Example: test x1 effect (holding x2), then test x2 effect
summary(man, test = "Pillai")  # already gives term-by-term in most cases

# -----------------------------
# 5) Comparing to separate univariate regressions
# -----------------------------
fit_y1 <- lm(y1 ~ x1 + x2, data = dat)
fit_y2 <- lm(y2 ~ x1 + x2, data = dat)

summary(fit_y1)
summary(fit_y2)

# Note: point estimates match the columns of coef(fit),
# but multivariate inference (e.g., Pillai/Wilks) is joint across y1,y2.

# -----------------------------
# 6) Basic diagnostics
# -----------------------------
# Residual plots for each response
par(mfrow = c(2, 2))
plot(fitted(fit)[,1], R[,1], xlab = "Fitted y1", ylab = "Residual y1",
     main = "Residuals vs Fitted (y1)")
abline(h = 0)

qqnorm(R[,1], main = "QQ plot (y1 residuals)"); qqline(R[,1])

plot(fitted(fit)[,2], R[,2], xlab = "Fitted y2", ylab = "Residual y2",
     main = "Residuals vs Fitted (y2)")
abline(h = 0)

qqnorm(R[,2], main = "QQ plot (y2 residuals)"); qqline(R[,2])

par(mfrow = c(1, 1))

# Residual correlation (useful check when you expect correlated outcomes)
cor(R)

# -----------------------------
# 7) Optional: Build H and E matrices explicitly for testing a single predictor
# -----------------------------
# For intuition: H (hypothesis SSP) for testing x1 (1 df) can be computed by
# comparing reduced vs full models.

fit_full <- lm(cbind(y1, y2) ~ x1 + x2, data = dat)
fit_red  <- lm(cbind(y1, y2) ~ x2, data = dat)   # remove x1

E_full <- t(residuals(fit_full)) %*% residuals(fit_full)
E_red  <- t(residuals(fit_red))  %*% residuals(fit_red)

H_x1 <- E_red - E_full     # for nested models, H = E_reduced - E_full
E_x1 <- E_full

H_x1
E_x1

# From H and E you can compute eigenvalues of solve(E) %*% H, then stats.
eig <- eigen(solve(E_x1, H_x1), only.values = TRUE)$values
eig <- Re(eig)             # numerical noise can introduce tiny imaginary parts
eig
