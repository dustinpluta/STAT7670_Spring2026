# ============================================================
# PCA for dealing with multicollinearity: Principal Components Regression (PCR)
# - Simulate highly collinear predictors
# - Show instability in OLS (large SEs, high VIFs)
# - Fit PCR by regressing y on first K PCs (orthogonal predictors)
# - Compare coefficient stability and predictive performance (CV)
# ============================================================

set.seed(7670)

# ----------------------------
# 1) Simulate multicollinear predictors
# ----------------------------
n <- 250
p <- 6

# Create two latent factors that drive most predictors -> strong collinearity
z1 <- rnorm(n)
z2 <- rnorm(n)

X <- cbind(
  1.0*z1 + 0.1*rnorm(n),
  0.9*z1 + 0.1*rnorm(n),
  0.8*z1 + 0.1*rnorm(n),
  1.0*z2 + 0.1*rnorm(n),
  0.9*z2 + 0.1*rnorm(n),
  0.7*z2 + 0.1*rnorm(n)
)
colnames(X) <- paste0("X", 1:p)

# True model uses one variable from each latent block
beta_true <- c(2.0, 0, 0, -1.5, 0, 0)
y <- as.numeric(X %*% beta_true + rnorm(n, sd = 1.0))

df <- data.frame(y, X)

# ----------------------------
# 2) OLS shows multicollinearity symptoms
# ----------------------------
ols <- lm(y ~ ., data = df)
summary(ols)

# VIFs (requires car)
if (!requireNamespace("car", quietly = TRUE)) install.packages("car")
library(car)

vifs <- vif(ols)
vifs

# A quick look at predictor correlations
round(cor(df[, -1]), 2)

# ----------------------------
# 3) PCR: PCA on X, then regress y on first K PCs
# ----------------------------
# Standardize predictors before PCA (typical for PCR)
X_scaled <- scale(df[, -1], center = TRUE, scale = TRUE)

pca <- prcomp(X_scaled, center = FALSE, scale. = FALSE) # already scaled

# Scree: variance explained by PCs
eig <- pca$sdev^2
pve <- eig / sum(eig)
plot(pve, type = "b",
     xlab = "PC", ylab = "Proportion of variance explained",
     main = "PCR: Variance explained by PCs")
plot(cumsum(pve), type = "b", ylim = c(0, 1),
     xlab = "Number of PCs", ylab = "Cumulative proportion",
     main = "PCR: Cumulative variance explained")
abline(h = c(0.8, 0.9, 0.95), lty = 2)

# Choose K (for demo, try K = 2 since data are driven by 2 latent factors)
K <- 2

# Construct PC score data frame for regression
scores <- as.data.frame(pca$x)
colnames(scores) <- paste0("PC", 1:p)

pcr_fit <- lm(y ~ ., data = cbind(y = df$y, scores[, 1:K, drop = FALSE]))
summary(pcr_fit)

# Note: VIFs on PCs are ~1 because PCs are orthogonal
vif(pcr_fit)

# ----------------------------
# 4) Convert PCR coefficients back to original X scale (interpretability)
# ----------------------------
# Model in PC space:
#   y = a0 + sum_{k=1}^K a_k * PC_k
# with PC = X_scaled %*% V   (V = loadings/rotation)
#
# So coefficients on X_scaled are: b_scaled = V[,1:K] %*% a[1:K]
# Then convert from scaled X to original X:
#   y = intercept + sum_j b_j * X_j
# where b_j = b_scaled_j / sd_j
# and intercept = a0 - sum_j b_j * mean_j

a <- coef(pcr_fit)
a0 <- a[1]
aK <- a[-1]

V <- pca$rotation                 # p x p
b_scaled <- as.vector(V[, 1:K, drop = FALSE] %*% aK)

x_means <- attr(X_scaled, "scaled:center")
x_sds   <- attr(X_scaled, "scaled:scale")

b_original <- b_scaled / x_sds
intercept_original <- a0 - sum(b_original * x_means)

pcr_coef_original_scale <- c("(Intercept)" = intercept_original,
                             setNames(b_original, colnames(X)))
round(pcr_coef_original_scale, 3)

# Compare to OLS coefficients (often unstable when predictors are collinear)
round(coef(ols), 3)

# ----------------------------
# 5) Compare predictive performance with cross-validation
# ----------------------------
# We'll do a simple train/test split and compare test MSE for:
# - OLS
# - PCR with K chosen by CV
set.seed(7670)
idx <- sample.int(n, size = round(0.7 * n))
train <- df[idx, ]
test  <- df[-idx, ]

# OLS on training data
ols_tr <- lm(y ~ ., data = train)
pred_ols <- predict(ols_tr, newdata = test)
mse_ols <- mean((test$y - pred_ols)^2)

# CV to choose K for PCR (manual K-fold CV)
set.seed(7670)
Kmax <- p
folds <- sample(rep(1:5, length.out = nrow(train)))

Xtr <- scale(train[, -1], center = TRUE, scale = TRUE)
ytr <- train$y

pca_tr <- prcomp(Xtr, center = FALSE, scale. = FALSE)
scores_tr <- pca_tr$x

cv_mse <- rep(NA_real_, Kmax)

for (k in 1:Kmax) {
  fold_mse <- rep(NA_real_, 5)
  for (f in 1:5) {
    tr_idx <- which(folds != f)
    va_idx <- which(folds == f)
    
    fit_k <- lm(ytr[tr_idx] ~ scores_tr[tr_idx, 1:k, drop = FALSE])
    pred_k <- predict(fit_k,
                      newdata = data.frame(scores_tr[va_idx, 1:k, drop = FALSE]))
    fold_mse[f] <- mean((ytr[va_idx] - pred_k)^2)
  }
  cv_mse[k] <- mean(fold_mse)
}

plot(1:Kmax, cv_mse, type = "b",
     xlab = "Number of PCs (K)", ylab = "5-fold CV MSE",
     main = "Choose K for PCR by CV")
K_star <- which.min(cv_mse)
K_star

# Fit PCR with chosen K* on training set
fit_pcr_star <- lm(ytr ~ scores_tr[, 1:K_star, drop = FALSE])

# Prepare test PC scores using *training* centering/scaling and training loadings
Xte <- scale(test[, -1],
             center = attr(Xtr, "scaled:center"),
             scale  = attr(Xtr, "scaled:scale"))

scores_te <- Xte %*% pca_tr$rotation  # n_test x p

pred_pcr <- as.numeric(
  coef(fit_pcr_star)[1] + scores_te[, 1:K_star, drop = FALSE] %*% coef(fit_pcr_star)[-1]
)
mse_pcr <- mean((test$y - pred_pcr)^2)

# Report
c(MSE_OLS = mse_ols, MSE_PCR = mse_pcr, K_star = K_star)

# ----------------------------
# Interpretation notes (for class)
# ----------------------------
# - Multicollinearity inflates OLS standard errors and makes coefficients unstable.
# - PCR replaces correlated predictors with orthogonal PCs -> VIFs ~ 1.
# - Keeping only first K PCs can reduce variance (stabilize estimates) but may add bias.
# - Choose K by CV or by a variance-explained rule, depending on objective (prediction vs explanation).
