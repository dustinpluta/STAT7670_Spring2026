# ============================================================
# Canonical Correlation Analysis (CCA) — Lecture Demo
# Dataset: iris
# ============================================================

# ------------------------------------------------------------
# 0. Setup
# ------------------------------------------------------------
# Goal of this demo:
# Show how CCA finds relationships between TWO sets of variables

# In this example:
#   X = sepal measurements
#   Y = petal measurements

# Key question:
# "What linear combination of sepals is most strongly associated
#  with what linear combination of petals?"

data(iris)

# Always inspect the data first
head(iris)
str(iris)

# ------------------------------------------------------------
# 1. Define the two blocks
# ------------------------------------------------------------

# Block 1: Sepal variables
X_raw <- iris[, c("Sepal.Length", "Sepal.Width")]

# Block 2: Petal variables
Y_raw <- iris[, c("Petal.Length", "Petal.Width")]

# Important teaching point:
# CCA is about RELATIONSHIP BETWEEN BLOCKS,
# not prediction and not classification.

# ------------------------------------------------------------
# 2. Standardize variables
# ------------------------------------------------------------

# CCA is sensitive to scale.
# We standardize so that all variables contribute comparably.

X <- scale(X_raw)
Y <- scale(Y_raw)

# Verify standardization
colMeans(X)
apply(X, 2, sd)

colMeans(Y)
apply(Y, 2, sd)

# ------------------------------------------------------------
# 3. Explore correlations
# ------------------------------------------------------------

cat("\nWithin sepal block:\n")
print(cor(X))

cat("\nWithin petal block:\n")
print(cor(Y))

cat("\nBetween blocks (sepals vs petals):\n")
print(cor(X, Y))

# Teaching note:
# Students should see that there ARE cross-block correlations,
# which motivates using CCA.

# ------------------------------------------------------------
# 4. Fit CCA
# ------------------------------------------------------------

cca_fit <- cancor(X, Y)

cat("\nCanonical correlations:\n")
print(cca_fit$cor)

# Interpretation:
# These are the maximum correlations achievable between
# linear combinations of X and Y.

# Usually:
# - First correlation is large
# - Second is much smaller

cat("\nCanonical coefficients for sepals (X):\n")
print(cca_fit$xcoef)

cat("\nCanonical coefficients for petals (Y):\n")
print(cca_fit$ycoef)

# Teaching note:
# These are weights defining the canonical variates.
# But raw coefficients are often hard to interpret directly.

# ------------------------------------------------------------
# 5. Compute canonical variates
# ------------------------------------------------------------

U <- X %*% cca_fit$xcoef   # canonical variates for X
V <- Y %*% cca_fit$ycoef   # canonical variates for Y

colnames(U) <- c("U1", "U2")
colnames(V) <- c("V1", "V2")

# ------------------------------------------------------------
# 6. Verify key CCA properties
# ------------------------------------------------------------

cat("\nCorrelation between canonical variates:\n")
print(cor(U, V))

cat("\nWithin U (should be ~uncorrelated):\n")
print(cor(U))

cat("\nWithin V (should be ~uncorrelated):\n")
print(cor(V))

# Teaching note:
# Key properties to emphasize:
# - cor(U1, V1) = first canonical correlation
# - cor(U2, V2) = second canonical correlation
# - canonical variates within each block are uncorrelated

# ------------------------------------------------------------
# 7. Plot first canonical pair
# ------------------------------------------------------------

plot(U[, 1], V[, 1],
     col = as.numeric(iris$Species),
     pch = 19,
     xlab = "U1 (sepals)",
     ylab = "V1 (petals)",
     main = "First canonical pair")

legend("topleft",
       legend = levels(iris$Species),
       col = 1:3,
       pch = 19)

abline(lm(V[, 1] ~ U[, 1]), lty = 2)

# Teaching note:
# This plot shows the strongest relationship between the two blocks.
# If correlation is high, points lie close to a line.

# ------------------------------------------------------------
# 8. Plot second canonical pair
# ------------------------------------------------------------

plot(U[, 2], V[, 2],
     col = as.numeric(iris$Species),
     pch = 19,
     xlab = "U2 (sepals)",
     ylab = "V2 (petals)",
     main = "Second canonical pair")

legend("topleft",
       legend = levels(iris$Species),
       col = 1:3,
       pch = 19)

abline(lm(V[, 2] ~ U[, 2]), lty = 2)

# Teaching note:
# This pair captures remaining association after removing the first.
# Usually much weaker.

# ------------------------------------------------------------
# 9. Interpret using loadings (IMPORTANT)
# ------------------------------------------------------------

# Loadings = correlation between original variables and canonical variates

x_loadings <- cor(X, U)
y_loadings <- cor(Y, V)

cat("\nLoadings for sepal variables:\n")
print(x_loadings)

cat("\nLoadings for petal variables:\n")
print(y_loadings)

# Teaching note:
# These are MUCH easier to interpret than raw coefficients.

# Example interpretation:
# If all variables load strongly on U1/V1,
# then the first canonical dimension represents "overall size".

# ------------------------------------------------------------
# 10. Cross-loadings
# ------------------------------------------------------------

cat("\nCross-loadings: sepal variables vs V:\n")
print(cor(X, V))

cat("\nCross-loadings: petal variables vs U:\n")
print(cor(Y, U))

# Teaching note:
# These show how one block relates to the canonical variates of the other block.

# ------------------------------------------------------------
# 11. Group means (optional but insightful)
# ------------------------------------------------------------

cca_scores <- data.frame(U1 = U[, 1],
                         U2 = U[, 2],
                         V1 = V[, 1],
                         V2 = V[, 2],
                         Species = iris$Species)

cat("\nSpecies means in canonical space:\n")
print(aggregate(cbind(U1, V1) ~ Species,
                data = cca_scores,
                mean))

# Teaching note:
# Even though CCA does NOT use species,
# the structure we find may still reflect species differences.

# ------------------------------------------------------------
# 12. Final interpretation summary
# ------------------------------------------------------------

cat("\nInterpretation summary:\n")
cat("CCA identifies latent dimensions linking sepals and petals.\n")
cat("The first canonical pair captures the dominant shared pattern.\n")
cat("Loadings indicate which variables drive this relationship.\n")
cat("Subsequent pairs capture weaker, residual associations.\n")

# ============================================================
# End of demo
# ============================================================