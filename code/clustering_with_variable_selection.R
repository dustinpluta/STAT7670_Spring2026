# ============================================================
# Clustering with Many Irrelevant Variables:
# Raw Data vs Variable Selection
# ============================================================
#
# Goal:
# Show that clustering can degrade when many irrelevant variables
# are included, and improve after removing those variables.
#
# Methods compared:
#   1. Hierarchical clustering (Ward)
#   2. Gaussian mixture model (mclust)
#
# Evaluation:
#   Adjusted Rand Index (ARI)
#
# ============================================================

# install.packages("mclust")   # run once if needed
library(mclust)

set.seed(123)

# ------------------------------------------------------------
# 1. Simulate data
# ------------------------------------------------------------

n_per <- 100
K <- 3
n <- K * n_per

# Three clusters defined by only first two variables
centers <- rbind(
  c(-3,  0),
  c( 3,  0),
  c( 0,  4)
)

signal <- NULL
truth <- NULL

for (k in 1:K) {
  Xk <- cbind(
    rnorm(n_per, centers[k,1], 0.8),
    rnorm(n_per, centers[k,2], 0.8)
  )
  signal <- rbind(signal, Xk)
  truth <- c(truth, rep(k, n_per))
}

truth <- factor(truth)

# Add many irrelevant variables
p_noise <- 50

noise <- matrix(
  rnorm(n * p_noise, mean = 0, sd = 1),
  nrow = n,
  ncol = p_noise
)

# Full dataset: 2 informative + 50 irrelevant
X_full <- cbind(signal, noise)
colnames(X_full) <- c("x1", "x2", paste0("noise", 1:p_noise))

# Standardize variables
X_full <- scale(X_full)

# ------------------------------------------------------------
# 2. Raw clustering using all variables
# ------------------------------------------------------------

# Hierarchical clustering
hc_raw <- hclust(dist(X_full), method = "ward.D2")
cl_hc_raw <- cutree(hc_raw, k = K)

# GMM clustering
gmm_raw <- Mclust(X_full, G = K, verbose = FALSE)
cl_gmm_raw <- gmm_raw$classification

# ------------------------------------------------------------
# 3. Variable selection
# ------------------------------------------------------------
# In this simulation we KNOW x1 and x2 are the informative vars.
# In practice, selection would use domain knowledge or screening.

X_sel <- X_full[, c("x1", "x2")]

# ------------------------------------------------------------
# 4. Clustering after variable selection
# ------------------------------------------------------------

# Hierarchical clustering
hc_sel <- hclust(dist(X_sel), method = "ward.D2")
cl_hc_sel <- cutree(hc_sel, k = K)

# GMM clustering
gmm_sel <- Mclust(X_sel, G = K, verbose = FALSE)
cl_gmm_sel <- gmm_sel$classification

# ------------------------------------------------------------
# 5. Compare using Adjusted Rand Index
# ------------------------------------------------------------

results <- data.frame(
  Method = c(
    "Hierarchical (all variables)",
    "Hierarchical (selected vars)",
    "GMM (all variables)",
    "GMM (selected vars)"
  ),
  ARI = c(
    adjustedRandIndex(truth, cl_hc_raw),
    adjustedRandIndex(truth, cl_hc_sel),
    adjustedRandIndex(truth, cl_gmm_raw),
    adjustedRandIndex(truth, cl_gmm_sel)
  )
)

print(results)

# ------------------------------------------------------------
# 6. Visualize true structure using selected variables
# ------------------------------------------------------------

par(mfrow = c(2,2))

plot(X_sel,
     col = truth,
     pch = 19,
     xlab = "x1",
     ylab = "x2",
     main = "True Clusters")

plot(X_sel,
     col = cl_hc_raw,
     pch = 19,
     xlab = "x1",
     ylab = "x2",
     main = "Hierarchical: All Variables")

plot(X_sel,
     col = cl_hc_sel,
     pch = 19,
     xlab = "x1",
     ylab = "x2",
     main = "Hierarchical: After Selection")

plot(X_sel,
     col = cl_gmm_sel,
     pch = 19,
     xlab = "x1",
     ylab = "x2",
     main = "GMM: After Selection")

par(mfrow = c(1,1))

# ------------------------------------------------------------
# 7. Dendrogram comparison
# ------------------------------------------------------------

par(mfrow = c(1,2))

plot(hc_raw,
     labels = FALSE,
     main = "Ward Dendrogram (All Variables)")

rect.hclust(hc_raw, k = K, border = 2:4)

plot(hc_sel,
     labels = FALSE,
     main = "Ward Dendrogram (Selected Variables)")

rect.hclust(hc_sel, k = K, border = 2:4)

par(mfrow = c(1,1))

# ------------------------------------------------------------
# 8. Interpretation summary
# ------------------------------------------------------------

cat("\n====================================================\n")
cat("Interpretation Summary\n")
cat("====================================================\n")
cat("1. The true cluster structure depends only on x1 and x2.\n")
cat("2. Adding many irrelevant variables distorts distances and\n")
cat("   can reduce clustering performance.\n")
cat("3. After removing irrelevant variables, ARI often improves.\n")
cat("4. Variable selection is often more effective than generic\n")
cat("   dimension reduction when many predictors are irrelevant.\n")