############################################################
# Application-style dataset + full multivariate regression analysis in R
#
# Setting: Adolescent pulmonary function study
# Responses (multivariate, correlated):
#   - FEV1 (L): forced expiratory volume in 1s
#   - FVC  (L): forced vital capacity
#   - PEF  (L/s): peak expiratory flow
#
# Predictors:
#   - age (years), height (cm), sex (0/1), smoker (0/1), activity (hrs/week)
#
# Workflow:
#   1) Simulate a realistic dataset with correlated outcomes
#   2) EDA (summaries + plots)
#   3) Fit multivariate multiple regression: lm(cbind(...) ~ ...)
#   4) Hypothesis testing (Pillai, Wilks) via manova()
#   5) Follow-up univariate models + multiplicity control
#   6) Diagnostics:
#       - leverage (hat values)
#       - multivariate residual Mahalanobis distances + chi-square QQ
#       - response-wise residual plots + QQ plots
#       - influence: leave-one-out change in Wilks' Lambda
############################################################

set.seed(123)

## ----------------------------
## 1) Create application-style dataset
## ----------------------------
n <- 220

age      <- runif(n, 12, 18)                         # years
height   <- rnorm(n, mean = 155 + 6*(age-12), sd=6)  # cm, increasing with age
sex      <- rbinom(n, 1, 0.5)                        # 1 = male, 0 = female
smoker   <- rbinom(n, 1, plogis(-6 + 0.35*(age-12))) # smoking increases with age
activity <- pmax(0, rnorm(n, mean = 5 + 1.2*sex, sd = 2))  # hrs/wk

# Design matrix (including intercept)
X <- model.matrix(~ age + height + sex + smoker + activity)

# True coefficients for three correlated outcomes (q = 3)
# Columns correspond to: FEV1, FVC, PEF
B <- rbind(
  c(-4.0,  -4.2,  -8.0),    # intercept (units: L, L, L/s)
  c( 0.06,  0.07,  0.10),   # age
  c( 0.030, 0.035, 0.060),  # height
  c( 0.12,  0.15,  0.40),   # sex (male higher on average)
  c(-0.18, -0.15, -0.35),   # smoker (lower pulmonary function)
  c( 0.01,  0.01,  0.03)    # activity (small positive effect)
)

# Error covariance among outcomes (correlated physiology)
Sigma <- matrix(
  c(0.10^2, 0.10*0.11*0.75, 0.10*0.30*0.55,
    0.10*0.11*0.75, 0.11^2, 0.11*0.30*0.60,
    0.10*0.30*0.55, 0.11*0.30*0.60, 0.30^2),
  nrow = 3, byrow = TRUE
)

# Generate multivariate normal errors without extra packages
Z <- matrix(rnorm(n*3), n, 3)
E <- Z %*% chol(Sigma)

Y <- X %*% B + E
colnames(Y) <- c("FEV1", "FVC", "PEF")

dat <- data.frame(
  FEV1 = Y[,1], FVC = Y[,2], PEF = Y[,3],
  age = age, height = height, sex = factor(sex, labels = c("Female","Male")),
  smoker = factor(smoker, labels = c("No","Yes")),
  activity = activity
)

# Add a couple of diagnostic "stress tests"
# (1) a high-leverage X outlier
dat$height[7] <- dat$height[7] + 30
dat$age[7] <- dat$age[7] + 2
# (2) a multivariate Y outlier
dat$FEV1[19] <- dat$FEV1[19] + 0.8
dat$FVC[19]  <- dat$FVC[19]  - 0.7
dat$PEF[19]  <- dat$PEF[19]  + 1.2

## ----------------------------
## 2) Quick EDA
## ----------------------------
cat("\n--- EDA summaries ---\n")
print(summary(dat[, c("FEV1","FVC","PEF","age","height","activity","sex","smoker")]))

cat("\nOutcome correlations:\n")
print(cor(dat[, c("FEV1","FVC","PEF")]))

op <- par(mfrow = c(2,2), mar = c(4,4,2,1))
pairs(dat[, c("FEV1","FVC","PEF")], main = "Pulmonary outcomes (pairs plot)")
plot(dat$height, dat$FEV1, xlab="Height (cm)", ylab="FEV1 (L)", main="FEV1 vs Height")
plot(dat$age, dat$FEV1, xlab="Age (years)", ylab="FEV1 (L)", main="FEV1 vs Age")
boxplot(FEV1 ~ smoker, data=dat, ylab="FEV1 (L)", main="FEV1 by Smoking")
par(op)

## ----------------------------
## 3) Fit multivariate multiple regression
## ----------------------------
fit <- lm(cbind(FEV1, FVC, PEF) ~ age + height + sex + smoker + activity, data = dat)
man <- manova(fit)

cat("\n--- Multivariate hypothesis tests (global, per term) ---\n")
print(summary(man, test = "Pillai"))  # robust
print(summary(man, test = "Wilks"))   # common

## Interpretation reminder:
## - A significant term indicates it shifts the *mean response vector*,
##   i.e., affects at least one linear combination of (FEV1, FVC, PEF).

## ----------------------------
## 4) Follow-up univariate regressions (with multiplicity control)
## ----------------------------
cat("\n--- Follow-up univariate regressions ---\n")
fits_uni <- list(
  FEV1 = lm(FEV1 ~ age + height + sex + smoker + activity, data=dat),
  FVC  = lm(FVC  ~ age + height + sex + smoker + activity, data=dat),
  PEF  = lm(PEF  ~ age + height + sex + smoker + activity, data=dat)
)

# Extract p-values for the smoking effect across outcomes
p_smoke <- sapply(fits_uni, function(m) summary(m)$coefficients["smokerYes","Pr(>|t|)"])
cat("\nUnivariate p-values for smoking effect:\n")
print(p_smoke)
cat("\nMultiplicity-adjusted (Holm) p-values for smoking effect:\n")
print(p.adjust(p_smoke, method = "holm"))

# Optional: view full summaries
# lapply(fits_uni, summary)

## ----------------------------
## 5) Diagnostics: leverage (X-space)
## ----------------------------
Xhat <- model.matrix(fit)
H <- Xhat %*% solve(t(Xhat) %*% Xhat) %*% t(Xhat)
hii <- diag(H)
lev_thresh <- 2 * ncol(Xhat) / n

cat("\n--- Leverage diagnostics ---\n")
cat("Leverage threshold 2*(p+1)/n =", round(lev_thresh, 3), "\n")
cat("Top 8 leverage observations:\n")
print(head(sort(hii, decreasing = TRUE), 8))

## ----------------------------
## 6) Diagnostics: multivariate residual distances (Mahalanobis)
## ----------------------------
Ehat <- residuals(fit)  # n x 3
df_resid <- fit$df.residual
Sigma_hat <- crossprod(Ehat) / df_resid
Sinv <- solve(Sigma_hat)
D2 <- rowSums((Ehat %*% Sinv) * Ehat)  # diag(Ehat %*% Sinv %*% t(Ehat))

chi_cut_975 <- qchisq(0.975, df = 3)
flag_md <- which(D2 > chi_cut_975)

cat("\n--- Multivariate residual outliers ---\n")
cat("Chi-square(0.975, df=3) cutoff =", round(chi_cut_975, 3), "\n")
cat("Flagged observations (D^2 > cutoff):\n")
print(flag_md)

# QQ plot for D^2 against chi-square(3)
D2_sorted <- sort(D2)
theo <- qchisq(ppoints(n), df = 3)

plot(theo, D2_sorted,
     xlab = "Theoretical quantiles (Chi-square df=3)",
     ylab = "Ordered residual Mahalanobis D^2",
     main = "QQ: Multivariate residual distances")
abline(0, 1, lty = 2)

## ----------------------------
## 7) Diagnostics: response-wise residual plots and QQ
## ----------------------------
Yhat <- fitted(fit)

op <- par(mfrow = c(3,2), mar = c(4,4,2,1))
resp_names <- c("FEV1","FVC","PEF")

for (j in 1:3) {
  plot(Yhat[,j], Ehat[,j],
       xlab = paste0("Fitted ", resp_names[j]),
       ylab = "Residual",
       main = paste0("Residuals vs fitted: ", resp_names[j]))
  abline(h = 0, lty = 2)
  
  qqnorm(Ehat[,j], main = paste0("QQ residuals: ", resp_names[j]))
  qqline(Ehat[,j], lty = 2)
}
par(op)

## ----------------------------
## 8) Influence: leave-one-out change in Wilks' Lambda
## ----------------------------
get_wilks <- function(manova_obj) {
  tab <- summary(manova_obj, test="Wilks")$stats
  tab[, "Wilks"]
}

wilks_full <- get_wilks(man)

Delta <- rep(NA_real_, n)
for (i in 1:n) {
  dat_i <- dat[-i, ]
  fit_i <- lm(cbind(FEV1, FVC, PEF) ~ age + height + sex + smoker + activity, data = dat_i)
  man_i <- manova(fit_i)
  wilks_i <- get_wilks(man_i)
  
  common <- intersect(names(wilks_full), names(wilks_i))
  Delta[i] <- max(abs(wilks_full[common] - wilks_i[common]))
}

cat("\n--- Influence (max |Delta Wilks| across terms) ---\n")
print(head(sort(Delta, decreasing = TRUE), 10))

plot(hii, Delta,
     xlab = "Leverage h_ii",
     ylab = "Max |Delta Wilks| (leave-one-out)",
     main = "Influence on multivariate inference vs leverage")
abline(v = lev_thresh, lty = 2)

## ----------------------------
## 9) Sensitivity analysis (optional but recommended)
## ----------------------------
# If you have flagged points, rerun without them and compare multivariate tests.
flag_union <- sort(unique(c(which(hii > lev_thresh), flag_md, which(Delta > quantile(Delta, 0.99)))))

cat("\nFlag union (high leverage OR large D^2 OR top 1% influence):\n")
print(flag_union)

if (length(flag_union) > 0 && length(flag_union) < n-10) {
  dat_clean <- dat[-flag_union, ]
  fit_clean <- lm(cbind(FEV1, FVC, PEF) ~ age + height + sex + smoker + activity, data = dat_clean)
  man_clean <- manova(fit_clean)
  
  cat("\n--- Multivariate tests (original) ---\n")
  print(summary(man, test="Pillai"))
  cat("\n--- Multivariate tests (after removing flagged points) ---\n")
  print(summary(man_clean, test="Pillai"))
}

############################################################
# What you should report (typical write-up structure)
#
# 1) Describe outcomes and predictors; note outcomes are correlated.
# 2) Fit multivariate multiple regression; report Pillai (and/or Wilks)
#    for each predictor (global tests).
# 3) For significant predictors, present follow-up univariate effects
#    with multiplicity control (e.g., Holm).
# 4) Diagnostics:
#    - identify high-leverage points
#    - identify multivariate residual outliers via Mahalanobis distance
#    - check marginal residual plots/QQ
#    - run sensitivity analysis excluding influential points
############################################################
