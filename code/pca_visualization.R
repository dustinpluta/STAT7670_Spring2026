# ============================================================
# PCA for high-dimensional visualization (simulated data)
# - Simulate p-dimensional data with known group structure
# - Run PCA and visualize PC1 vs PC2 (and optionally PC3)
# - Show how variance explained matters for interpretation
# ============================================================

set.seed(7670)

# ----------------------------
# 1) Simulate high-dimensional data with group structure
# ----------------------------
n_per_group <- 80
G <- 3
n <- n_per_group * G
p <- 60  # high dimension

group <- factor(rep(paste0("G", 1:G), each = n_per_group))

# Create a low-dimensional "signal" subspace of dimension q
q <- 3

# Latent scores for each observation (q-dimensional)
Z <- matrix(rnorm(n * q), nrow = n, ncol = q)

# Add group mean shifts in latent space (creates separable structure)
mu_latent <- rbind(
  c( 2.0,  0.0,  0.0),
  c( 0.0,  2.0,  0.0),
  c( 0.0,  0.0,  2.0)
)
Z <- Z + mu_latent[as.integer(group), ]

# Map latent signal into p-dimensional observed space via a random loading matrix
A <- matrix(rnorm(p * q), nrow = p, ncol = q)

# Noise level controls how visible separation is in PCs
sigma_eps <- 1.0
E <- matrix(rnorm(n * p, sd = sigma_eps), nrow = n, ncol = p)

# Observed data matrix (n x p)
X <- Z %*% t(A) + E
colnames(X) <- paste0("X", 1:p)

# ----------------------------
# 2) Run PCA
# ----------------------------
# Standardize since variables are arbitrary / on comparable footing
pca <- prcomp(X, center = TRUE, scale. = TRUE)

# Variance explained
eig <- pca$sdev^2
pve <- eig / sum(eig)

# Scree plot + cumulative variance explained
par(mfrow = c(1, 2))

plot(pve, type = "b",
     xlab = "Principal Component",
     ylab = "Proportion variance explained",
     main = "Scree: Proportion variance explained")

plot(cumsum(pve), type = "b", ylim = c(0, 1),
     xlab = "Number of PCs",
     ylab = "Cumulative variance explained",
     main = "Cumulative variance explained")
abline(h = c(0.7, 0.8, 0.9, 0.95), lty = 2)

par(mfrow = c(1, 1))

# ----------------------------
# 3) High-dim visualization: score plots
# ----------------------------
scores <- as.data.frame(pca$x)

# A simple color palette that doesn't require extra packages
cols <- setNames(c("black", "blue", "red"), levels(group))

plot(scores$PC1, scores$PC2,
     col = cols[group], pch = 19,
     xlab = paste0("PC1 (", round(100 * pve[1], 1), "%)"),
     ylab = paste0("PC2 (", round(100 * pve[2], 1), "%)"),
     main = "High-dimensional visualization: PC1 vs PC2")
legend("topright", legend = levels(group), col = cols, pch = 19, bty = "n")

# Optional: PC1 vs PC3 (sometimes separation appears off the PC1-PC2 plane)
plot(scores$PC1, scores$PC3,
     col = cols[group], pch = 19,
     xlab = paste0("PC1 (", round(100 * pve[1], 1), "%)"),
     ylab = paste0("PC3 (", round(100 * pve[3], 1), "%)"),
     main = "High-dimensional visualization: PC1 vs PC3")
legend("topright", legend = levels(group), col = cols, pch = 19, bty = "n")

# ----------------------------
# 4) A quick diagnostic: how well does 2D preserve group separation?
# ----------------------------
# Fit a simple LDA-like rule in PC space using nearest-centroid classification
# (No extra packages: classify by closest group centroid in (PC1, PC2).)

PC12 <- as.matrix(scores[, c("PC1", "PC2")])
centroids <- do.call(rbind, lapply(levels(group), function(g) colMeans(PC12[group == g, , drop = FALSE])))
rownames(centroids) <- levels(group)

pred <- apply(PC12, 1, function(u) {
  d <- rowSums((centroids - matrix(u, nrow = nrow(centroids), ncol = 2, byrow = TRUE))^2)
  rownames(centroids)[which.min(d)]
})

acc <- mean(pred == as.character(group))
acc

# ----------------------------
# Teaching notes (what to point out)
# ----------------------------
# - PCA is unsupervised: it does not use the group labels.
# - Separation in PC plots indicates that between-group variability aligns with high-variance directions.
# - Always report variance explained on axes; "no separation" may mean the separation is in PC3+.
# - If sigma_eps is increased, separation will degrade; if decreased, separation strengthens.
