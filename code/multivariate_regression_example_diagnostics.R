############################################################
# Multivariate multiple regression diagnostics (R example)
# - Simulate data with 3 correlated responses (q = 3)
# - Fit multivariate multiple regression via lm(cbind(... ) ~ ...)
# - Run multivariate diagnostics:
#   (1) leverage / hat values (from X)
#   (2) residual Mahalanobis distances + chi-square QQ plot
#   (3) univariate residual checks per response (fitted vs residuals, QQ)
#   (4) influence via leave-one-out change in Wilks' Lambda (global test)
############################################################

set.seed(1)

## ----------------------------
## 1) Simulate multivariate data
## ----------------------------
n <- 180
p <- 3          # predictors: x1, x2, x3 (plus intercept)
q <- 3          # number of responses

x1 <- rnorm(n)
x2 <- rnorm(n)
x3 <- rbinom(n, 1, 0.4)  # binary predictor

X <- model.matrix(~ x1 + x2 + x3)  # n x (p+1)

# True coefficient matrix B: (p+1) x q
B_true <- rbind(
  c( 0.0,  0.5, -0.3),   # intercept effects on each response
  c( 1.0,  0.8,  0.0),   # x1 effects
  c(-0.8,  0.2,  0.6),   # x2 effects
  c( 0.0,  0.9, -0.7)    # x3 effects
)

# Correlated error covariance Sigma (q x q)
Sigma <- matrix(
  c(1.0, 0.6, 0.3,
    0.6, 1.0, 0.5,
    0.3, 0.5, 1.0),
  nrow = q, byrow = TRUE
)

# Draw multivariate normal errors without extra packages:
# E = Z %*% chol(Sigma) where Z ~ N(0, I)
Z <- matrix(rnorm(n * q), n, q)
E <- Z %*% chol(Sigma)

# Construct responses: Y = X B + E
Y <- X %*% B_true + E
colnames(Y) <- c("y1", "y2", "y3")

dat <- data.frame(y1 = Y[,1], y2 = Y[,2], y3 = Y[,3], x1 = x1, x2 = x2, x3 = x3)

## Add a couple of problematic points (to make diagnostics interesting)
# (a) a high-leverage point in X-space
dat$x1[5] <- 4.5
dat$x2[5] <- -4.0
# (b) a multivariate outlier in Y-space
dat$y1[12] <- dat$y1[12] + 6
dat$y2[12] <- dat$y2[12] - 6
dat$y3[12] <- dat$y3[12] + 6

## ----------------------------
## 2) Fit multivariate multiple regression
## ----------------------------
fit <- lm(cbind(y1, y2, y3) ~ x1 + x2 + x3, data = dat)

# Global multivariate tests for each predictor (Wilks, Pillai, etc.)
man <- manova(fit)
print(summary(man, test = "Pillai"))
print(summary(man, test = "Wilks"))

## ----------------------------
## 3) Leverage / hat values (depends only on X)
## ----------------------------
# Use the model matrix from the fitted object to compute hat diag: H = X (X'X)^{-1} X'
Xhat <- model.matrix(fit)  # n x (p+1)
H <- Xhat %*% solve(t(Xhat) %*% Xhat) %*% t(Xhat)
hii <- diag(H)

# Common heuristic threshold for "high leverage"
# (There are different conventions; this is a standard starting point.)
lev_thresh <- 2 * ncol(Xhat) / n
cat("\nLeverage threshold (2*(p+1)/n) =", round(lev_thresh, 3), "\n")
cat("Top 6 leverage observations:\n")
print(head(sort(hii, decreasing = TRUE), 6))

## ----------------------------
## 4) Multivariate residual distances (Mahalanobis)
## ----------------------------
# Residual matrix (n x q)
Ehat <- residuals(fit)

# Estimate Sigma via residual crossproduct / df
df_resid <- fit$df.residual
Sigma_hat <- crossprod(Ehat) / df_resid

# Mahalanobis distance for each residual vector:
# D_i^2 = e_i' Sigma^{-1} e_i
Sinv <- solve(Sigma_hat)
D2 <- rowSums((Ehat %*% Sinv) * Ehat)  # efficient diagonal of Ehat %*% Sinv %*% t(Ehat)

# Compare to chi-square(q) reference (approx)
chi_cut_975 <- qchisq(0.975, df = q)
cat("\nChi-square(0.975, df=q) cutoff =", round(chi_cut_975, 3), "\n")

# Flag suspicious large residual distances
flag_md <- which(D2 > chi_cut_975)
cat("Observations with D^2 > chi-square(0.975):\n")
print(flag_md)

## QQ plot of D^2 vs chi-square(q)
# If errors are MVN and the model is appropriate, points should lie near a line.
D2_sorted <- sort(D2)
theo <- qchisq(ppoints(n), df = q)

plot(theo, D2_sorted,
     main = "QQ plot: residual Mahalanobis D^2 vs Chi-square(q)",
     xlab = "Theoretical quantiles (Chi-square)",
     ylab = "Ordered residual D^2")
abline(0, 1, lty = 2)

## ----------------------------
## 5) Response-wise residual plots (useful, but not sufficient alone)
## ----------------------------
Yhat <- fitted(fit)

op <- par(mfrow = c(3, 2), mar = c(4, 4, 2, 1))

for (j in 1:q) {
  # Fitted vs residuals for response j
  plot(Yhat[, j], Ehat[, j],
       xlab = paste0("Fitted values (y", j, ")"),
       ylab = paste0("Residuals (y", j, ")"),
       main = paste0("Residuals vs Fitted: y", j))
  abline(h = 0, lty = 2)
  
  # Normal QQ plot for residuals of response j (marginal check)
  qqnorm(Ehat[, j], main = paste0("QQ plot residuals: y", j))
  qqline(Ehat[, j], lty = 2)
}

par(op)

## ----------------------------
## 6) Influence on global multivariate inference (leave-one-out)
## ----------------------------
# One concrete influence measure: how much Wilks' Lambda for (x1, x2, x3) changes
# when you delete observation i.
#
# We'll compute:
#   Delta_i = Wilks_full - Wilks_(leave-one-out)
#
# Larger |Delta_i| suggests stronger influence on the multivariate test.

get_wilks_for_predictors <- function(fit_manova) {
  # summary.manova returns a table with rows = terms, including x1, x2, x3
  tab <- summary(fit_manova, test = "Wilks")$stats
  # tab columns include: Df, Wilks, approx F, num Df, den Df, Pr(>F)
  # We'll focus on the overall Wilks for each term; you could also track Pillai, etc.
  out <- tab[, "Wilks"]
  return(out)
}

wilks_full <- get_wilks_for_predictors(man)

Delta <- rep(NA_real_, n)
for (i in 1:n) {
  dat_i <- dat[-i, ]
  fit_i <- lm(cbind(y1, y2, y3) ~ x1 + x2 + x3, data = dat_i)
  man_i <- manova(fit_i)
  wilks_i <- get_wilks_for_predictors(man_i)
  
  # Compare the *vector* of Wilks stats for x1,x2,x3; summarize by max abs change
  common_terms <- intersect(names(wilks_full), names(wilks_i))
  Delta[i] <- max(abs(wilks_full[common_terms] - wilks_i[common_terms]))
}

# Show top potentially influential points under this metric
cat("\nTop 10 observations by max |Delta Wilks| across predictors:\n")
print(head(sort(Delta, decreasing = TRUE), 10))

# Plot influence metric vs leverage to separate X-outliers vs Y-outliers
plot(hii, Delta,
     xlab = "Leverage h_ii",
     ylab = "Max |Delta Wilks| (leave-one-out)",
     main = "Influence on multivariate test vs leverage")
abline(v = lev_thresh, lty = 2)
text(hii, Delta, labels = ifelse(Delta > quantile(Delta, 0.98), seq_len(n), ""),
     pos = 3, cex = 0.7)

## ----------------------------
## 7) What to look for (interpretation guide)
## ----------------------------
cat("\nInterpretation guide:\n")
cat("1) High leverage points: hii >> 2*(p+1)/n can dominate coefficient estimates.\n")
cat("2) Large Mahalanobis D^2: suggests unusual *joint* residual patterns.\n")
cat("3) Response-wise residual plots: reveal nonlinearity/heteroskedasticity per outcome.\n")
cat("4) Large leave-one-out Delta Wilks: suggests sensitivity of global inference to specific observations.\n")
