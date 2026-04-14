# ============================================================
# Full k-means clustering example in R using USArrests
# ============================================================

# Goal:
# Cluster U.S. states based on crime-related variables:
# - Murder
# - Assault
# - UrbanPop
# - Rape
#
# This demo shows:
# 1. exploratory analysis
# 2. why scaling matters
# 3. fitting k-means
# 4. choosing K with elbow and silhouette methods
# 5. interpreting the resulting clusters
# 6. plotting clusters in PCA space
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
# 2. Basic exploratory analysis
# ------------------------------------------------------------

# Look at pairwise relationships
pairs(USArrests,
      pch = 19,
      main = "USArrests: pairwise plots")

# Correlation matrix
cor_mat <- cor(USArrests)
print(cor_mat)

# ------------------------------------------------------------
# 3. Why scaling matters
# ------------------------------------------------------------
# k-means uses Euclidean distance, so variables on larger scales
# can dominate the clustering. We therefore standardize first.

X <- scale(USArrests)

# Verify standardization
colMeans(X)
apply(X, 2, sd)

# ------------------------------------------------------------
# 4. Choose number of clusters K
# ------------------------------------------------------------

# ---- 4a. Elbow method ----

set.seed(123)

K_max <- 10
wss <- numeric(K_max)

for (k in 1:K_max) {
  km_tmp <- kmeans(X, centers = k, nstart = 25)
  wss[k] <- km_tmp$tot.withinss
}

plot(1:K_max, wss,
     type = "b",
     pch = 19,
     xlab = "Number of clusters K",
     ylab = "Total within-cluster sum of squares",
     main = "Elbow plot for k-means")

# ---- 4b. Silhouette method ----
# install.packages("cluster")  # run once if needed
library(cluster)

avg_sil <- numeric(K_max)

for (k in 2:K_max) {
  km_tmp <- kmeans(X, centers = k, nstart = 25)
  sil <- silhouette(km_tmp$cluster, dist(X))
  avg_sil[k] <- mean(sil[, "sil_width"])
}

plot(2:K_max, avg_sil[2:K_max],
     type = "b",
     pch = 19,
     xlab = "Number of clusters K",
     ylab = "Average silhouette width",
     main = "Silhouette method for k-means")

# Choose a value of K based on the plots.
# For this example, we will use K = 3.
# You can change this if your plots suggest another choice.

# ------------------------------------------------------------
# 5. Fit k-means clustering
# ------------------------------------------------------------

set.seed(123)

k <- 3
km_fit <- kmeans(X, centers = k, nstart = 50)

print(km_fit)

# Cluster assignments
cluster_assignments <- km_fit$cluster
table(cluster_assignments)

# Add cluster labels to data frame
df$Cluster <- factor(cluster_assignments)

# ------------------------------------------------------------
# 6. Examine cluster centers
# ------------------------------------------------------------
# km_fit$centers are in standardized units

cat("\nCluster centers in standardized units:\n")
print(round(km_fit$centers, 2))

# More interpretable: compute cluster means in original units
cluster_summary <- aggregate(USArrests,
                             by = list(Cluster = df$Cluster),
                             FUN = mean)

cat("\nCluster means in original units:\n")
print(round(cluster_summary, 2))

# Cluster sizes
cat("\nCluster sizes:\n")
print(table(df$Cluster))

# ------------------------------------------------------------
# 7. Inspect which states belong to each cluster
# ------------------------------------------------------------

states_by_cluster <- split(df$State, df$Cluster)

cat("\nStates by cluster:\n")
print(states_by_cluster)

# ------------------------------------------------------------
# 8. Visualize clusters using PCA
# ------------------------------------------------------------
# Since the data are 4-dimensional, PCA gives a convenient
# 2D summary for plotting.

pca_fit <- prcomp(X, center = FALSE, scale. = FALSE)

# Scores on the first two PCs
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
     main = "k-means clusters visualized in PCA space")

text(pca_scores$PC1, pca_scores$PC2,
     labels = pca_scores$State,
     pos = 3,
     cex = 0.7)

legend("topright",
       legend = levels(pca_scores$Cluster),
       col = 1:k,
       pch = 19,
       title = "Cluster")

# ------------------------------------------------------------
# 9. Plot cluster profiles
# ------------------------------------------------------------

# Transpose cluster centers in original units for plotting
cluster_means_mat <- as.matrix(cluster_summary[, -1])
rownames(cluster_means_mat) <- paste("Cluster", cluster_summary$Cluster)

matplot(t(cluster_means_mat),
        type = "b",
        pch = 19,
        lty = 1,
        xaxt = "n",
        xlab = "Variable",
        ylab = "Cluster mean",
        main = "Cluster profiles (original scale)")

axis(1, at = 1:ncol(cluster_means_mat), labels = colnames(cluster_means_mat))
legend("topright",
       legend = rownames(cluster_means_mat),
       col = 1:k,
       lty = 1,
       pch = 19)

# ------------------------------------------------------------
# 10. Within-cluster quality diagnostics
# ------------------------------------------------------------

cat("\nTotal within-cluster sum of squares:\n")
print(km_fit$tot.withinss)

cat("\nBetween-cluster sum of squares:\n")
print(km_fit$betweenss)

cat("\nProportion of variance explained by clustering:\n")
print(km_fit$betweenss / km_fit$totss)

# Silhouette plot for the chosen clustering
sil_fit <- silhouette(km_fit$cluster, dist(X))
plot(sil_fit, main = "Silhouette plot for chosen k-means solution")

cat("\nAverage silhouette width for chosen solution:\n")
print(mean(sil_fit[, "sil_width"]))

# ------------------------------------------------------------
# 11. Compare multiple random starts
# ------------------------------------------------------------
# k-means can converge to local minima, so nstart matters.

set.seed(123)
km_nstart1  <- kmeans(X, centers = k, nstart = 1)
km_nstart50 <- kmeans(X, centers = k, nstart = 50)

cat("\nTotal within-cluster SS with nstart = 1:\n")
print(km_nstart1$tot.withinss)

cat("\nTotal within-cluster SS with nstart = 50:\n")
print(km_nstart50$tot.withinss)

# ------------------------------------------------------------
# 12. Order states by cluster and print a compact table
# ------------------------------------------------------------

results_table <- df[, c("State", "Cluster", "Murder", "Assault", "UrbanPop", "Rape")]
results_table <- results_table[order(results_table$Cluster), ]

cat("\nStates ordered by cluster:\n")
print(results_table)

# ------------------------------------------------------------
# 13. Interpretation summary
# ------------------------------------------------------------

cat("\n====================================================\n")
cat("Interpretation Summary\n")
cat("====================================================\n")
cat("1. k-means partitions the states into clusters based on similar crime profiles.\n")
cat("2. Because distance is used, scaling the variables is essential.\n")
cat("3. The cluster means help interpret the types of states in each group.\n")
cat("4. PCA provides a useful low-dimensional visualization of the clustering.\n")
cat("5. The elbow and silhouette methods help guide the choice of K, but the decision is not automatic.\n")