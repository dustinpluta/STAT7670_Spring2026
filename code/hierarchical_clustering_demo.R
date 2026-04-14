# ============================================================
# Full Hierarchical Clustering Demo in R using USArrests
# ============================================================
#
# Goal:
# Use hierarchical clustering to group U.S. states based on:
# - Murder
# - Assault
# - UrbanPop
# - Rape
#
# This demo shows:
# 1. exploratory analysis
# 2. why scaling matters
# 3. hierarchical clustering with multiple linkage methods
# 4. dendrogram interpretation
# 5. choosing a number of clusters
# 6. comparing cluster assignments
# 7. visualizing clusters in PCA space
#
# ============================================================

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

cor_mat <- cor(USArrests)
cat("\nCorrelation matrix:\n")
print(round(cor_mat, 3))

# ------------------------------------------------------------
# 3. Standardize the variables
# ------------------------------------------------------------
# Important because hierarchical clustering depends on distance,
# and the variables are on different scales.

X <- scale(USArrests)

cat("\nColumn means after scaling:\n")
print(round(colMeans(X), 5))

cat("\nColumn SDs after scaling:\n")
print(round(apply(X, 2, sd), 5))

# ------------------------------------------------------------
# 4. Compute distance matrix
# ------------------------------------------------------------

d <- dist(X, method = "euclidean")

cat("\nDistance matrix summary:\n")
print(summary(as.vector(d)))

# ------------------------------------------------------------
# 5. Fit hierarchical clustering using different linkage methods
# ------------------------------------------------------------

hc_complete <- hclust(d, method = "complete")
hc_average  <- hclust(d, method = "average")
hc_single   <- hclust(d, method = "single")
hc_ward     <- hclust(d, method = "ward.D2")

# ------------------------------------------------------------
# 6. Plot dendrograms
# ------------------------------------------------------------

par(mfrow = c(2, 2))

plot(hc_complete,
     labels = rownames(USArrests),
     main = "Complete Linkage",
     xlab = "",
     sub = "",
     cex = 0.7)

plot(hc_average,
     labels = rownames(USArrests),
     main = "Average Linkage",
     xlab = "",
     sub = "",
     cex = 0.7)

plot(hc_single,
     labels = rownames(USArrests),
     main = "Single Linkage",
     xlab = "",
     sub = "",
     cex = 0.7)

plot(hc_ward,
     labels = rownames(USArrests),
     main = "Ward's Method",
     xlab = "",
     sub = "",
     cex = 0.7)

par(mfrow = c(1, 1))

# Teaching note:
# Ward's method often gives compact, balanced clusters.
# Single linkage often shows chaining.

# ------------------------------------------------------------
# 7. Focus on Ward's method for interpretation
# ------------------------------------------------------------

plot(hc_ward,
     labels = rownames(USArrests),
     main = "Hierarchical Clustering of USArrests (Ward's Method)",
     xlab = "",
     sub = "",
     cex = 0.7)

# Add rectangles for a chosen number of clusters
k <- 4
rect.hclust(hc_ward, k = k, border = 2:5)

# ------------------------------------------------------------
# 8. Cut the dendrogram into clusters
# ------------------------------------------------------------

clusters_ward <- cutree(hc_ward, k = k)

cat("\nCluster sizes (Ward, k = 4):\n")
print(table(clusters_ward))

# Add cluster assignments to data frame
df$Cluster <- factor(clusters_ward)

# ------------------------------------------------------------
# 9. Inspect which states fall into each cluster
# ------------------------------------------------------------

states_by_cluster <- split(df$State, df$Cluster)

cat("\nStates by cluster:\n")
print(states_by_cluster)

# ------------------------------------------------------------
# 10. Cluster summaries in original units
# ------------------------------------------------------------

cluster_summary <- aggregate(USArrests,
                             by = list(Cluster = df$Cluster),
                             FUN = mean)

cat("\nCluster means in original units:\n")
print(round(cluster_summary, 2))

# ------------------------------------------------------------
# 11. Compare cluster profiles
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
# 12. Visualize clusters in PCA space
# ------------------------------------------------------------
# PCA is used only for 2D visualization, not for clustering.

pca_fit <- prcomp(X, center = FALSE, scale. = FALSE)

pca_scores <- data.frame(
  PC1 = pca_fit$x[, 1],
  PC2 = pca_fit$x[, 2],
  Cluster = df$Cluster,
  State = df$State
)

plot(pca_scores$PC1, pca_scores$PC2,
     col = as.numeric(pca_scores$Cluster),
     pch = 19,
     xlab = "PC1",
     ylab = "PC2",
     main = "Ward Clusters Visualized in PCA Space")

text(pca_scores$PC1, pca_scores$PC2,
     labels = pca_scores$State,
     pos = 3,
     cex = 0.65)

legend("topright",
       legend = levels(pca_scores$Cluster),
       col = 1:k,
       pch = 19,
       title = "Cluster")

# ------------------------------------------------------------
# 13. Compare hierarchical cluster assignments across linkage methods
# ------------------------------------------------------------

clusters_complete <- cutree(hc_complete, k = k)
clusters_average  <- cutree(hc_average,  k = k)
clusters_single   <- cutree(hc_single,   k = k)

cat("\nComparison: Ward vs Complete\n")
print(table(Ward = clusters_ward, Complete = clusters_complete))

cat("\nComparison: Ward vs Average\n")
print(table(Ward = clusters_ward, Average = clusters_average))

cat("\nComparison: Ward vs Single\n")
print(table(Ward = clusters_ward, Single = clusters_single))

# ------------------------------------------------------------
# 14. Optional: silhouette analysis for Ward clusters
# ------------------------------------------------------------

# install.packages("cluster")   # run once if needed
library(cluster)

sil_ward <- silhouette(clusters_ward, d)

plot(sil_ward,
     main = "Silhouette Plot for Ward Clustering")

cat("\nAverage silhouette width:\n")
print(mean(sil_ward[, "sil_width"]))

# ------------------------------------------------------------
# 15. Examine different numbers of clusters
# ------------------------------------------------------------

k_values <- 2:6
avg_sil <- numeric(length(k_values))

for (i in seq_along(k_values)) {
  cl_tmp <- cutree(hc_ward, k = k_values[i])
  sil_tmp <- silhouette(cl_tmp, d)
  avg_sil[i] <- mean(sil_tmp[, "sil_width"])
}

plot(k_values, avg_sil,
     type = "b",
     pch = 19,
     xlab = "Number of clusters k",
     ylab = "Average silhouette width",
     main = "Choosing k for Ward Clustering")

# ------------------------------------------------------------
# 16. Ordered results table
# ------------------------------------------------------------

results_table <- df[, c("State", "Cluster", "Murder", "Assault", "UrbanPop", "Rape")]
results_table <- results_table[order(results_table$Cluster), ]

cat("\nStates ordered by cluster:\n")
print(results_table)

# ------------------------------------------------------------
# 17. Interpretation summary
# ------------------------------------------------------------

cat("\n====================================================\n")
cat("Interpretation Summary\n")
cat("====================================================\n")
cat("1. Hierarchical clustering builds a nested grouping structure.\n")
cat("2. Ward's method tends to produce compact clusters and is often\n")
cat("   a good default choice.\n")
cat("3. The dendrogram helps visualize how states merge into larger groups.\n")
cat("4. Cutting the dendrogram at k =", k, "produces", k, "clusters.\n")
cat("5. Cluster summaries in the original variables help interpret the\n")
cat("   substantive meaning of each group.\n")
cat("6. PCA is useful for visualizing the resulting clusters in two dimensions.\n")

# ------------------------------------------------------------
# 18. Optional discussion prompt
# ------------------------------------------------------------

cat("\nSuggested discussion questions:\n")
cat("- Which linkage method seems most interpretable here?\n")
cat("- Do the clusters appear stable across linkage methods?\n")
cat("- How do the resulting state groups differ in crime profile?\n")
cat("- How might the conclusions change if the variables were not scaled?\n")