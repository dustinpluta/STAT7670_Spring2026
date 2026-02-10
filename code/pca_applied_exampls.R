# ============================================================
# PCA coding examples in R using built-in datasets:
#   1) iris
#   2) USArrests
#   3) mtcars
#
# Focus:
# - Standard preprocessing
# - Scree plots
# - Variance explained
# - Loadings and score plots
# ============================================================

# ------------------------------------------------------------
# 1) PCA on iris
# ------------------------------------------------------------
data(iris)

# Use only numeric variables
X_iris <- iris[, 1:4]

# PCA on correlation matrix (standardized variables)
pca_iris <- prcomp(X_iris, center = TRUE, scale. = TRUE)

# Summary
summary(pca_iris)

# Scree plot
plot(pca_iris$sdev^2, type = "b",
     xlab = "Principal Component",
     ylab = "Eigenvalue",
     main = "Iris: Scree Plot")

# Scores plot (colored by species)
scores_iris <- pca_iris$x
plot(scores_iris[,1], scores_iris[,2],
     col = as.numeric(iris$Species),
     pch = 19,
     xlab = "PC1", ylab = "PC2",
     main = "Iris: PC1 vs PC2")
legend("topright", legend = levels(iris$Species),
       col = 1:3, pch = 19)

# Loadings
round(pca_iris$rotation[, 1:2], 3)

# ------------------------------------------------------------
# 2) PCA on USArrests
# ------------------------------------------------------------
data(USArrests)

# PCA on covariance matrix (no scaling)
pca_usa_cov <- prcomp(USArrests, center = TRUE, scale. = FALSE)

# PCA on correlation matrix (with scaling)
pca_usa_cor <- prcomp(USArrests, center = TRUE, scale. = TRUE)

# Compare variance explained
summary(pca_usa_cov)
summary(pca_usa_cor)

# Scree plots
par(mfrow = c(1, 2))

plot(pca_usa_cov$sdev^2, type = "b",
     xlab = "PC",
     ylab = "Eigenvalue",
     main = "USArrests: Covariance PCA")

plot(pca_usa_cor$sdev^2, type = "b",
     xlab = "PC",
     ylab = "Eigenvalue",
     main = "USArrests: Correlation PCA")

par(mfrow = c(1, 1))

# Scores plot (correlation PCA)
scores_usa <- pca_usa_cor$x
plot(scores_usa[,1], scores_usa[,2],
     xlab = "PC1", ylab = "PC2",
     main = "USArrests: PC1 vs PC2")
text(scores_usa[,1], scores_usa[,2],
     labels = rownames(USArrests),
     cex = 0.6, pos = 3)

# Loadings
round(pca_usa_cor$rotation[, 1:2], 3)

# ------------------------------------------------------------
# 3) PCA on mtcars
# ------------------------------------------------------------
data(mtcars)

# PCA on correlation matrix
pca_mtcars <- prcomp(mtcars, center = TRUE, scale. = TRUE)

# Summary
summary(pca_mtcars)

# Scree plot
plot(pca_mtcars$sdev^2, type = "b",
     xlab = "Principal Component",
     ylab = "Eigenvalue",
     main = "mtcars: Scree Plot")

# Scores plot
scores_mtcars <- pca_mtcars$x
plot(scores_mtcars[,1], scores_mtcars[,2],
     xlab = "PC1", ylab = "PC2",
     main = "mtcars: PC1 vs PC2")
text(scores_mtcars[,1], scores_mtcars[,2],
     labels = rownames(mtcars),
     cex = 0.6, pos = 3)

# Loadings
round(pca_mtcars$rotation[, 1:2], 3)

# ------------------------------------------------------------
# Notes for instruction
# ------------------------------------------------------------
# - iris: clean structure, excellent for first PCA and visualization
# - USArrests: ideal for demonstrating the impact of scaling
# - mtcars: realistic applied example with less interpretable PCs
# - prcomp() uses SVD; eigenvalues are pca$sdev^2
# - Scores (pca$x) are used for dimension reduction
# - Loadings (pca$rotation) explain component structure
