set.seed(1)

# -----------------------------
# Simulate MANOVA data
# -----------------------------
n_per_group <- 40
g <- 3
p <- 3

group <- factor(rep(1:g, each = n_per_group))

# Group means: small/moderate multivariate separation
mu <- rbind(
  c(0.0,  0.0,  0.0),
  c(0.4,  0.2, -0.1),
  c(0.7,  0.3,  0.2)
)

# Common covariance (positive definite)
Sigma <- matrix(c(
  1.0, 0.4, 0.2,
  0.4, 1.0, 0.3,
  0.2, 0.3, 1.0
), nrow = p, byrow = TRUE)

rmvnorm_chol <- function(n, mu, Sigma) {
  L <- chol(Sigma)
  Z <- matrix(rnorm(n * length(mu)), nrow = n)
  sweep(Z %*% L, 2, mu, "+")
}

Y <- do.call(rbind, lapply(1:g, function(k) rmvnorm_chol(n_per_group, mu[k, ], Sigma)))
colnames(Y) <- c("y1", "y2", "y3")

dat <- data.frame(group = group, Y)

# -----------------------------
# Fit one-way MANOVA
# -----------------------------
fit <- manova(cbind(y1, y2, y3) ~ group, data = dat)

# Compare omnibus test statistics
summary(fit, test = "Pillai")
summary(fit, test = "Wilks")
summary(fit, test = "Hotelling-Lawley")
summary(fit, test = "Roy")


tests <- c("Pillai", "Wilks", "Hotelling-Lawley", "Roy")

out <- lapply(tests, function(tt) {
  s <- summary(fit, test = tt)$stats
  # Keep the row for 'group' (the effect); column names vary slightly by test,
  # but generally include approx F, num Df, den Df, and Pr(>F).
  s_group <- s["group", , drop = FALSE]
  data.frame(
    test = tt,
    stat = unname(s_group[1]),
    approx_F = unname(s_group[2]),
    num_df = unname(s_group[3]),
    den_df = unname(s_group[4]),
    p_value = unname(s_group[5]),
    row.names = NULL
  )
})

results <- do.call(rbind, out)
print(results)


summary.aov(fit)  # separate ANOVA per response, no multiplicity control
