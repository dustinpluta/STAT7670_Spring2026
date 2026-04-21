# ============================================================
# Full Gaussian Mixture Model (GMM) Demo in R using mclust
# Example: USArrests data
# ============================================================
#
# Goal:
# Show how to use model-based clustering via Gaussian mixture models.
#
# This demo illustrates:
# 1. loading and exploring the data
# 2. standardizing variables
# 3. fitting Gaussian mixture models with mclust
# 4. selecting the number of clusters and covariance structure via BIC
# 5. examining cluster assignments and uncertainty
# 6. interpreting cluster profiles
# 7. visualizing the fitted clustering in PCA space
#
# ============================================================

# ------------------------------------------------------------
# 0. Install/load package
# ------------------------------------------------------------

# install.packages("mclust")   # run once if needed
library(mclust)

# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------

data("USArrests")

df <- USArrests
df$State <- rownames(USArrests)

head(df)
str(df)
summary(df)

# ------------------------------------------------------------
# 2. Exploratory analysis
# ------------------------------------------------------------

pairs(USArrests,
      pch = 19,
      main = "USArrests: Pairwise Plots")

cat("\nCorrelation matrix:\n")
print(round(cor(USArrests), 3))

# ------------------------------------------------------------
# 3. Standardize variables
# ------------------------------------------------------------
# Important because clustering depends on scale.

X <- scale(USArrests)

cat("\nMeans after scaling:\n")
print(round(colMeans(X), 5))

cat("\nStandard deviations after scaling:\n")
print(round(apply(X, 2, sd), 5))

# ------------------------------------------------------------
# 4. Fit Gaussian mixture models with mclust
# ------------------------------------------------------------
# Mclust automatically considers:
# - multiple numbers of clusters G
# - multiple covariance parameterizations
# and chooses the model with the best BIC

gmm_fit <- Mclust(X)

# Print summary
summary(gmm_fit)

# ------------------------------------------------------------
# 5. Inspect selected model
# ------------------------------------------------------------

cat("\nSelected model name:\n")
print(gmm_fit$modelName)

cat("\nSelected number of clusters:\n")
print(gmm_fit$G)

cat("\nMixing proportions:\n")
print(round(gmm_fit$parameters$pro, 3))

cat("\nCluster sizes:\n")
print(table(gmm_fit$classification))

# ------------------------------------------------------------
# 6. Plot BIC values
# ------------------------------------------------------------

plot(gmm_fit, what = "BIC")

# Teaching note:
# This plot helps show how mclust selected both:
# - number of mixture components
# - covariance structure

# ------------------------------------------------------------
# 7. Plot classification and uncertainty
# ------------------------------------------------------------

plot(gmm_fit, what = "classification")

plot(gmm_fit, what = "uncertainty")

# Teaching note:
# "classification" shows cluster assignments.
# "uncertainty" shows how ambiguous the assignments are.

# ------------------------------------------------------------
# 8. Cluster assignments and uncertainty
# ------------------------------------------------------------

df$Cluster <- factor(gmm_fit$classification)
df$Uncertainty <- gmm_fit$uncertainty

cat("\nFirst 10 states with cluster assignments and uncertainty:\n")
print(head(df[, c("State", "Cluster", "Uncertainty")], 10))

# States with highest uncertainty
uncertain_states <- df[order(-df$Uncertainty), c("State", "Cluster", "Uncertainty")]
cat("\nMost uncertain states:\n")
print(head(uncertain_states, 10))

# ------------------------------------------------------------
# 9. Posterior probabilities (soft assignments)
# ------------------------------------------------------------

post_probs <- gmm_fit$z

cat("\nPosterior probability matrix (first 10 observations):\n")
print(round(post_probs[1:10, ], 3))

# Add posterior probabilities to a table if desired
post_prob_df <- data.frame(
  State = df$State,
  Cluster = df$Cluster,
  Uncertainty = df$Uncertainty,
  post_probs
)

# ------------------------------------------------------------
# 10. Examine cluster means in original units
# ------------------------------------------------------------

cluster_summary <- aggregate(USArrests,
                             by = list(Cluster = df$Cluster),
                             FUN = mean)

cat("\nCluster means in original units:\n")
print(round(cluster_summary, 2))

# Cluster sizes
cat("\nCluster sizes:\n")
print(table(df$Cluster))

# States by cluster
states_by_cluster <- split(df$State, df$Cluster)

cat("\nStates by cluster:\n")
print(states_by_cluster)

# ------------------------------------------------------------
# 11. Visualize clusters in PCA space
# ------------------------------------------------------------

pca_fit <- prcomp(X, center = FALSE, scale. = FALSE)

pca_scores <- data.frame(
  PC1 = pca_fit$x[, 1],
  PC2 = pca_fit$x[, 2],
  Cluster = df$Cluster,
  State = df$State,
  Uncertainty = df$Uncertainty
)

plot(pca_scores$PC1, pca_scores$PC2,
     col = as.numeric(pca_scores$Cluster),
     pch = 19,
     xlab = "PC1",
     ylab = "PC2",
     main = "mclust Clusters in PCA Space")

text(pca_scores$PC1, pca_scores$PC2,
     labels = pca_scores$State,
     pos = 3,
     cex = 0.65)

legend("topright",
       legend = levels(pca_scores$Cluster),
       col = 1:length(levels(pca_scores$Cluster)),
       pch = 19,
       title = "Cluster")

# ------------------------------------------------------------
# 12. Plot cluster uncertainty in PCA space
# ------------------------------------------------------------

plot(pca_scores$PC1, pca_scores$PC2,
     cex = 1 + 3 * pca_scores$Uncertainty,
     col = as.numeric(pca_scores$Cluster),
     pch = 19,
     xlab = "PC1",
     ylab = "PC2",
     main = "Cluster Uncertainty in PCA Space")

text(pca_scores$PC1, pca_scores$PC2,
     labels = pca_scores$State,
     pos = 3,
     cex = 0.6)

legend("topright",
       legend = levels(pca_scores$Cluster),
       col = 1:length(levels(pca_scores$Cluster)),
       pch = 19,
       title = "Cluster")

# Teaching note:
# Larger points correspond to more uncertain classifications.

# ------------------------------------------------------------
# 13. Compare several candidate models manually
# ------------------------------------------------------------
# Optional: restrict the number of clusters considered

gmm_fit_small <- Mclust(X, G = 1:6)

cat("\nSelected model from G = 1:6:\n")
print(gmm_fit_small$modelName)

cat("\nSelected number of clusters from G = 1:6:\n")
print(gmm_fit_small$G)

plot(gmm_fit_small, what = "BIC")

# ------------------------------------------------------------
# 14. Compare mclust clustering to k-means
# ------------------------------------------------------------

set.seed(123)
km_fit <- kmeans(X, centers = gmm_fit$G, nstart = 25)

cat("\nContingency table: mclust vs k-means\n")
print(table(mclust = gmm_fit$classification,
            kmeans = km_fit$cluster))

# Teaching note:
# The methods may produce similar but not identical partitions because
# k-means is centroid-based while GMM clustering is probabilistic.

# ------------------------------------------------------------
# 15. Cluster profile plot
# ------------------------------------------------------------

cluster_means_mat <- as.matrix(cluster_summary[, -1])
rownames(cluster_means_mat) <- paste("Cluster", cluster_summary$Cluster)

matplot(t(cluster_means_mat),
        type = "b",
        pch = 19,
        lty = 1,
        xaxt = "n",
        xlab = "Variable",
        ylab = "Cluster mean",
        main = "Cluster Profiles (Original Scale)")

axis(1,
     at = 1:ncol(cluster_means_mat),
     labels = colnames(cluster_means_mat))

legend("topright",
       legend = rownames(cluster_means_mat),
       col = 1:nrow(cluster_means_mat),
       lty = 1,
       pch = 19)

# ------------------------------------------------------------
# 16. Ordered results table
# ------------------------------------------------------------

results_table <- df[, c("State", "Cluster", "Uncertainty",
                        "Murder", "Assault", "UrbanPop", "Rape")]
results_table <- results_table[order(results_table$Cluster, results_table$Uncertainty), ]

cat("\nStates ordered by cluster and uncertainty:\n")
print(results_table)

# ------------------------------------------------------------
# 17. Optional: classify new observations
# ------------------------------------------------------------
# We can use predict() on new standardized observations.

new_states <- data.frame(
  Murder   = c(3, 10, 15),
  Assault  = c(80, 180, 300),
  UrbanPop = c(45, 70, 80),
  Rape     = c(10, 20, 35)
)

# Standardize using original training means and sds
X_means <- attr(X, "scaled:center")
X_sds   <- attr(X, "scaled:scale")

new_states_scaled <- scale(new_states,
                           center = X_means,
                           scale = X_sds)

pred_new <- predict(gmm_fit, newdata = new_states_scaled)

cat("\nPredicted cluster for new observations:\n")
print(pred_new$classification)

cat("\nPosterior probabilities for new observations:\n")
print(round(pred_new$z, 3))

# ------------------------------------------------------------
# 18. Interpretation summary
# ------------------------------------------------------------

cat("\n====================================================\n")
cat("Interpretation Summary\n")
cat("====================================================\n")
cat("1. Gaussian mixture models treat the data as arising from a mixture\n")
cat("   of several Gaussian subpopulations.\n")
cat("2. mclust automatically chooses both the number of clusters and the\n")
cat("   covariance structure using BIC.\n")
cat("3. Unlike k-means, mclust provides soft assignments through posterior\n")
cat("   probabilities and a direct measure of uncertainty.\n")
cat("4. Cluster summaries in original units help interpret the types of\n")
cat("   states in each cluster.\n")
cat("5. Uncertainty is especially useful for states near cluster boundaries.\n")