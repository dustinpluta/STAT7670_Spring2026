# ============================================================
# Example: Why multivariate-response regression is different
# ============================================================

set.seed(123)

# -----------------------------
# 1) Data-generating mechanism
# -----------------------------
n <- 120
p <- 4
x <- rnorm(n)

# True effect: small but aligned across responses
beta <- 0.1

# Strong correlation between responses
Sigma <- matrix(c(1.0, 0.5, 0.35, 0.1,
                  0.5, 1.0, 0.85, 0.1,
                  0.35, 0.85, 1, 0.2,
                  0.1, 0.1, 0.2, 1), nrow = 4)

# Generate correlated errors
Z <- matrix(rnorm(n * p), n, p)
E <- Z %*% chol(Sigma)

# Responses
y1 <- beta * x + E[,1]
y2 <- beta * x + E[,2]
y3 <- beta * x + E[,3]
y4 <- beta * x + E[,4]

dat <- data.frame(y1 = y1, y2 = y2, y3 = y3, y4 = y4, x = x)

# -----------------------------
# 2) Separate univariate regressions
# -----------------------------
fit_y1 <- lm(y1 ~ x, data = dat)
fit_y2 <- lm(y2 ~ x, data = dat)
fit_y3 <- lm(y3 ~ x, data = dat)
fit_y4 <- lm(y4 ~ x, data = dat)

summary(fit_y1)
summary(fit_y2)
summary(fit_y3)
summary(fit_y4)

# Extract p-values
p_y1 <- summary(fit_y1)$coefficients["x", "Pr(>|t|)"]
p_y2 <- summary(fit_y2)$coefficients["x", "Pr(>|t|)"]
p_y3 <- summary(fit_y3)$coefficients["x", "Pr(>|t|)"]
p_y4 <- summary(fit_y4)$coefficients["x", "Pr(>|t|)"]

p_y1
p_y2
p_y3
p_y3
# Often: both p-values are marginal or not significant

# -----------------------------
# 3) Multivariate-response regression
# -----------------------------
fit_mv <- manova(cbind(y1, y2, y3, y4) ~ x, data = dat)

summary(fit_mv, test = "Wilks")
summary(fit_mv, test = "Pillai")

# -----------------------------
# 4) Explicit comparison
# -----------------------------
cat("\nUnivariate p-values:\n")
cat("  y1 ~ x:", round(p_y1, 4), "\n")
cat("  y2 ~ x:", round(p_y2, 4), "\n")
cat("  y3 ~ x:", round(p_y3, 4), "\n")
cat("  y4 ~ x:", round(p_y4, 4), "\n")

cat("\nMultivariate test p-values:\n")
cat("  Wilks:", summary(fit_mv, test = "Wilks")$stats[1, "Pr(>F)"], "\n")
cat("  Pillai:", summary(fit_mv, test = "Pillai")$stats[1, "Pr(>F)"], "\n")

# -----------------------------
# 5) Visualization for intuition
# # -----------------------------
# par(mfrow = c(1, 2))
# plot(x, y1, main = "y1 vs x (weak signal)", pch = 19)
# abline(fit_y1, col = "blue")
# plot(x, y2, main = "y2 vs x (weak signal)", pch = 19)
# abline(fit_y2, col = "blue")
# par(mfrow = c(1, 1))
