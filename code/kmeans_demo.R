# ============================================================
# k-means + k-NN Side-by-Side Demo in R
# ============================================================
#
# Goal:
# Show the difference between:
#
#   1. k-NN classification (supervised)
#   2. k-means clustering (unsupervised)
#
# using the same dataset.
#
# Dataset:
#   iris
#
# Key contrast:
# - k-NN uses the known species labels to classify
# - k-means ignores the species labels and tries to discover groups
#
# This demo includes:
# - exploratory plots
# - train/test k-NN classification
# - k-means clustering
# - comparison to true species labels
# - side-by-side visualizations
#
# ============================================================

# ------------------------------------------------------------
# 0. Install/load packages
# ------------------------------------------------------------

# install.packages("class")   # run once if needed
library(class)

# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------

data(iris)

head(iris)
str(iris)
table(iris$Species)

# Predictors
X <- iris[, 1:4]

# True labels
y <- iris$Species

# ------------------------------------------------------------
# 2. Standardize predictors
# ------------------------------------------------------------
# Important because both k-NN and k-means depend on distance.

X_scaled <- scale(X)

# ------------------------------------------------------------
# 3. Exploratory plots
# ------------------------------------------------------------

pairs(X,
      col = as.numeric(y),
      pch = 19,
      main = "Iris predictors colored by species")

legend("topright",
       legend = levels(y),
       col = 1:3,
       pch = 19,
       inset = 0.01)

# ------------------------------------------------------------
# 4. Two-dimensional visualization setup
# ------------------------------------------------------------
# For plots, use the petal variables because they separate well.

plot(X$Petal.Length, X$Petal.Width,
     col = as.numeric(y),
     pch = 19,
     xlab = "Petal Length",
     ylab = "Petal Width",
     main = "Iris data in petal space")

legend("topleft",
       legend = levels(y),
       col = 1:3,
       pch = 19)

# ------------------------------------------------------------
# 5. k-NN classification
# ------------------------------------------------------------
# This is supervised:
# we use the known species labels.

set.seed(123)

n <- nrow(X_scaled)
train_id <- sample(seq_len(n), size = round(0.7 * n))

X_train <- X_scaled[train_id, ]
X_test  <- X_scaled[-train_id, ]

y_train <- y[train_id]
y_test  <- y[-train_id]

# Choose k
k_knn <- 5

knn_pred <- knn(train = X_train,
                test = X_test,
                cl = y_train,
                k = k_knn)

cat("\n====================================\n")
cat("k-NN Classification Results\n")
cat("====================================\n")

cat("\nConfusion matrix:\n")
print(table(True = y_test, Predicted = knn_pred))

cat("\nTest accuracy:\n")
print(mean(knn_pred == y_test))

cat("\nTest misclassification rate:\n")
print(mean(knn_pred != y_test))

# ------------------------------------------------------------
# 6. k-NN on only the petal variables (for plotting regions)
# ------------------------------------------------------------

X2 <- scale(iris[, c("Petal.Length", "Petal.Width")])

X2_train <- X2[train_id, ]
X2_test  <- X2[-train_id, ]

knn_pred_2d <- knn(train = X2_train,
                   test = X2_test,
                   cl = y_train,
                   k = k_knn)

cat("\n2D k-NN test accuracy using petal variables only:\n")
print(mean(knn_pred_2d == y_test))

# ------------------------------------------------------------
# 7. Plot k-NN decision regions in 2D
# ------------------------------------------------------------

x1_seq <- seq(min(X2[, 1]) - 0.5, max(X2[, 1]) + 0.5, length.out = 200)
x2_seq <- seq(min(X2[, 2]) - 0.5, max(X2[, 2]) + 0.5, length.out = 200)

grid <- expand.grid(Petal.Length = x1_seq,
                    Petal.Width  = x2_seq)

grid_pred_knn <- knn(train = X2_train,
                     test = grid,
                     cl = y_train,
                     k = k_knn)

z_knn <- matrix(as.numeric(grid_pred_knn),
                nrow = length(x1_seq),
                ncol = length(x2_seq))

plot(X2_train[, 1], X2_train[, 2],
     col = as.numeric(y_train),
     pch = 19,
     xlab = "Scaled Petal Length",
     ylab = "Scaled Petal Width",
     main = "k-NN decision regions (training data)")

contour(x1_seq, x2_seq, z_knn,
        levels = c(1.5, 2.5),
        add = TRUE,
        drawlabels = FALSE,
        lwd = 2)

legend("topleft",
       legend = levels(y),
       col = 1:3,
       pch = 19)

# Teaching note:
# k-NN uses the known class labels and creates local classification regions.

# ------------------------------------------------------------
# 8. k-means clustering
# ------------------------------------------------------------
# This is unsupervised:
# species labels are NOT used in fitting.

set.seed(123)

k_clusters <- 3

km_fit <- kmeans(X_scaled, centers = k_clusters, nstart = 25)

cat("\n====================================\n")
cat("k-means Clustering Results\n")
cat("====================================\n")

cat("\nCluster sizes:\n")
print(km_fit$size)

cat("\nCluster centers (scaled variables):\n")
print(round(km_fit$centers, 3))

cat("\nWithin-cluster sum of squares:\n")
print(km_fit$withinss)

cat("\nTotal within-cluster sum of squares:\n")
print(km_fit$tot.withinss)

# ------------------------------------------------------------
# 9. Compare k-means clusters to true species
# ------------------------------------------------------------
# Since clustering labels are arbitrary, cluster 1/2/3 do not
# automatically correspond to setosa/versicolor/virginica.

cat("\nContingency table: true species vs k-means cluster\n")
print(table(TrueSpecies = y, Cluster = km_fit$cluster))

# ------------------------------------------------------------
# 10. Plot k-means clusters in 2D
# ------------------------------------------------------------

plot(X2[, 1], X2[, 2],
     col = km_fit$cluster,
     pch = 19,
     xlab = "Scaled Petal Length",
     ylab = "Scaled Petal Width",
     main = "k-means clusters in petal space")

legend("topleft",
       legend = paste("Cluster", 1:3),
       col = 1:3,
       pch = 19)

# Add cluster centers using the petal dimensions only
centers_2d <- km_fit$centers[, c("Petal.Length", "Petal.Width")]

points(centers_2d,
       col = 1:3,
       pch = 8,
       cex = 2,
       lwd = 2)

# ------------------------------------------------------------
# 11. Plot true species vs k-means clusters side-by-side
# ------------------------------------------------------------

par(mfrow = c(1, 2))

plot(X2[, 1], X2[, 2],
     col = as.numeric(y),
     pch = 19,
     xlab = "Scaled Petal Length",
     ylab = "Scaled Petal Width",
     main = "True species")

legend("topleft",
       legend = levels(y),
       col = 1:3,
       pch = 19)

plot(X2[, 1], X2[, 2],
     col = km_fit$cluster,
     pch = 19,
     xlab = "Scaled Petal Length",
     ylab = "Scaled Petal Width",
     main = "k-means clusters")

legend("topleft",
       legend = paste("Cluster", 1:3),
       col = 1:3,
       pch = 19)

par(mfrow = c(1, 1))

# Teaching note:
# This is a useful visual comparison:
# k-means can recover some of the true structure, but it does not use labels.

# ------------------------------------------------------------
# 12. Elbow plot for choosing K in k-means
# ------------------------------------------------------------

set.seed(123)

K_values <- 1:10
wss <- numeric(length(K_values))

for (k in K_values) {
  km_tmp <- kmeans(X_scaled, centers = k, nstart = 25)
  wss[k] <- km_tmp$tot.withinss
}

plot(K_values, wss,
     type = "b",
     pch = 19,
     xlab = "Number of clusters K",
     ylab = "Total within-cluster sum of squares",
     main = "Elbow plot for k-means")

# ------------------------------------------------------------
# 13. Optional: simple cluster-to-class mapping
# ------------------------------------------------------------
# This is not part of k-means itself, but it shows how well
# the clusters align with the known labels.

majority_label <- function(cluster_id, cluster_assignments, true_labels) {
  tab <- table(true_labels[cluster_assignments == cluster_id])
  names(which.max(tab))
}

cluster_to_species <- sapply(1:k_clusters,
                             majority_label,
                             cluster_assignments = km_fit$cluster,
                             true_labels = y)

cluster_to_species

km_species_pred <- factor(cluster_to_species[km_fit$cluster],
                          levels = levels(y))

cat("\nApproximate species assignment from k-means clusters:\n")
print(table(True = y, PredictedFromCluster = km_species_pred))

cat("\nApproximate clustering accuracy after majority-label matching:\n")
print(mean(km_species_pred == y))

# Teaching note:
# This gives a rough sense of alignment, but remember:
# clustering is unsupervised, so this is only a post hoc comparison.

# ------------------------------------------------------------
# 14. Side-by-side summary table
# ------------------------------------------------------------

summary_table <- data.frame(
  Method = c("k-NN classification", "k-means clustering"),
  Uses_Labels = c("Yes", "No"),
  Goal = c("Predict species", "Find groups"),
  Output = c("Predicted class", "Cluster assignment")
)

cat("\n====================================\n")
cat("Method Summary\n")
cat("====================================\n")
print(summary_table)

# ------------------------------------------------------------
# 15. Final teaching summary
# ------------------------------------------------------------

cat("\n====================================\n")
cat("Teaching Summary\n")
cat("====================================\n")
cat("1. k-NN is a supervised method: it uses known labels.\n")
cat("2. k-NN predicts the label of a point using nearby labeled points.\n")
cat("3. k-means is an unsupervised method: it ignores labels.\n")
cat("4. k-means groups observations by minimizing within-cluster variation.\n")
cat("5. On iris, k-means can recover some species structure, especially setosa,\n")
cat("   but it is not solving the same problem as k-NN.\n")