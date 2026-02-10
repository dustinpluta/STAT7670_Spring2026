# ============================================================
# PCA in R on simulated data
# Scree plot shown *before* running prcomp()
# (via eigenvalues of the sample covariance / correlation matrix)
# ============================================================

set.seed(7670)

# ----------------------------
# 1) Simulate data
# ----------------------------
n <- 300      # observations
p <- 8        # observed variables
k <- 2        # latent dimensions

# Latent factors
Z <- matrix(rnorm(n * k), nrow = n, ncol = k)

# Loading matrix
L <- matrix(0, nrow = p, ncol = k)
L[1:4, 1] <- c(1.2, 1.0, 0.9, 0.7)
L[5:8, 2] <- c(1.1, 1.0, 0.8, 0.6)
L[c(3, 6), ] <- L[c(3, 6), ] +
  matrix(c(0.3, 0.2,
           0.2, 0.3), nrow = 2, byrow = TRUE)

# Noise
sigma_eps <- 0.6
E <- matrix(rnorm(n * p, sd = sigma_eps), nrow = n, ncol = p)

# Observed data
X <- Z %*% t(L) + E
colnames(X) <- paste0("X", 1:p)
X_df <- as.data.frame(X)

# ----------------------------
# 2) Scree plot BEFORE PCA
# ----------------------------
# Decide whether we want PCA on covariance or correlation
# Here: correlation matrix (i.e., standardized variables)
X_scaled <- scale(X_df, center = TRUE, scale = TRUE)

# Sample correlation matrix
R <- cor(X_scaled)

# Eigenvalues of R
eig_pre <- eigen(R, symmetric = TRUE)$values

# Scree plot
plot(eig_pre, type = "b",
     xlab = "Component number",
     ylab = "Eigenvalue",
     main = "Scree Plot (from correlation matrix, before PCA)")
abline(h = 1, lty = 2)   # Kaiser rule reference

# Cumulative variance explained (pre-PCA)
pve_pre <- eig_pre / sum(eig_pre)
cum_pve_pre <- cumsum(pve_pre)

plot(cum_pve_pre, type = "b", ylim = c(0, 1),
     xlab = "Number of components",
     ylab = "Cumulative proportion of variance",
     main = "Cumulative Variance Explained (pre-PCA)")
abline(h = c(0.8, 0.9, 0.95), lty = 2)

# At this point, one would typically decide how many PCs to retain.

# ----------------------------
# 3) Now actually run PCA
# ----------------------------
pca <- prcomp(X_df, center = TRUE, scale. = TRUE)

# Summary
summary(pca)

# ----------------------------
# 4) Loadings and scores
# ----------------------------
# Loadings (rotation)
round(pca$rotation[, 1:2], 3)

# Scores
scores <- as.data.frame(pca$x)

plot(scores$PC1, scores$PC2,
     xlab = "PC1 score", ylab = "PC2 score",
     main = "Scores Plot: PC1 vs PC2")
abline(h = 0, v = 0, lty = 3)

# ----------------------------
# 5) Check consistency: eigenvalues vs prcomp
# ----------------------------
# Eigenvalues from prcomp (should match eig_pre)
eig_post <- pca$sdev^2

cbind(
  pre_PCA = round(eig_pre, 4),
  post_PCA = round(eig_post, 4)
)

# ----------------------------
# Notes
# ----------------------------
# - The scree plot does NOT require running PCA via prcomp().
#   It only requires the eigenvalues of the sample covariance or correlation matrix.
# - prcomp() is then used to obtain scores, loadings, and projections.
# - This mirrors how PCA is often motivated pedagogically:
#   (1) inspect eigenvalues → (2) choose dimension → (3) compute PCs.
