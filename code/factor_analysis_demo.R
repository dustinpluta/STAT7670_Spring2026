############################################################
# Factor Analysis (EFA) demo for graduate multivariate stats
# - Simulate data from a true common-factor model
# - Choose number of factors (parallel analysis)
# - Fit FA with different rotations
# - Interpret loadings, communalities, uniquenesses
# - Compute factor scores and do basic diagnostics
############################################################

set.seed(2026)

# Packages (install if needed)
pkgs <- c("MASS", "psych", "GPArotation")
to_install <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(to_install) > 0) install.packages(to_install)
lapply(pkgs, require, character.only = TRUE)

############################################################
# 1) Simulate data from a common-factor model
############################################################

n  <- 400   # sample size
p  <- 12    # observed variables
k  <- 3     # true number of common factors

# Construct a "simple structure" loading matrix Lambda (p x k)
Lambda <- matrix(0, nrow = p, ncol = k)
Lambda[1:4, 1]  <- c(.80, .75, .70, .65)  # Factor 1
Lambda[5:8, 2]  <- c(.85, .80, .70, .60)  # Factor 2
Lambda[9:12, 3] <- c(.80, .75, .70, .60)  # Factor 3

# Small cross-loadings to make it realistic
Lambda[2, 2] <- .15
Lambda[6, 1] <- .10
Lambda[10,2] <- .12

# Factor covariance (Phi). Orthogonal in this first run:
Phi <- diag(k)

# Unique variances (Psi): choose so total variances are ~1
communalities_true <- rowSums(Lambda %*% Phi * Lambda)
Psi <- diag(pmax(1 - communalities_true, 0.15))  # ensure not too tiny

# Generate latent factors and unique errors
F_latent <- MASS::mvrnorm(n, mu = rep(0, k), Sigma = Phi)
E_unique <- MASS::mvrnorm(n, mu = rep(0, p), Sigma = Psi)

# Observed data: X = F Lambda' + E
X <- F_latent %*% t(Lambda) + E_unique
colnames(X) <- paste0("x", 1:p)

# Standardize (common for FA when variables on different scales)
Xz <- scale(X)

############################################################
# 2) Quick checks: correlation matrix + KMO + Bartlett test
############################################################

R <- cor(Xz)

cat("\n--- KMO measure of sampling adequacy ---\n")
print(psych::KMO(R))

cat("\n--- Bartlett test of sphericity (H0: R = I) ---\n")
print(psych::cortest.bartlett(R, n = n))

############################################################
# 3) Choose number of factors: scree + parallel analysis
############################################################

cat("\n--- Parallel analysis (fa.parallel) ---\n")
psych::fa.parallel(Xz, fm = "ml", fa = "fa", n.iter = 50, main = "Parallel Analysis for FA")

# Suppose parallel analysis suggests m factors (students can decide).
# We'll fit m = 3 here (true k), but you can change this to what PA indicates.
m <- 3

############################################################
# 4) Fit FA (ML) with orthogonal rotation (varimax)
############################################################

fa_varimax <- psych::fa(Xz, nfactors = m, fm = "ml", rotate = "varimax", scores = "regression")

cat("\n=== FA (ML) with varimax rotation ===\n")
print(fa_varimax)

cat("\n--- Loadings (cut = .30) ---\n")
print(psych::print.psych(fa_varimax$loadings, cut = 0.30, sort = TRUE))

cat("\n--- Communalities (h2) and uniquenesses (u2) ---\n")
out1 <- data.frame(
  var = rownames(fa_varimax$loadings),
  h2  = round(fa_varimax$communality, 3),
  u2  = round(fa_varimax$uniquenesses, 3)
)
print(out1)

############################################################
# 5) Fit FA with oblique rotation (oblimin)
#    This is typically more realistic in social/biomedical data.
############################################################

fa_oblimin <- psych::fa(Xz, nfactors = m, fm = "ml", rotate = "oblimin", scores = "regression")

cat("\n=== FA (ML) with oblimin rotation ===\n")
print(fa_oblimin)

cat("\n--- Pattern loadings (cut = .30) ---\n")
print(psych::print.psych(fa_oblimin$loadings, cut = 0.30, sort = TRUE))

cat("\n--- Factor correlations (Phi) under oblique rotation ---\n")
print(round(fa_oblimin$Phi, 3))

############################################################
# 6) Model fit / residuals (one practical diagnostic)
############################################################

# Residual correlations: observed minus implied
resid_R <- fa_oblimin$residual
cat("\n--- Residual correlation summary ---\n")
cat("Max |residual|:", round(max(abs(resid_R)), 3), "\n")
cat("Mean |residual|:", round(mean(abs(resid_R[upper.tri(resid_R)])), 3), "\n")

# Look at the largest residual pairs (potential misfit or missing factor)
abs_res <- abs(resid_R)
abs_res[lower.tri(abs_res, diag = TRUE)] <- NA
top_idx <- order(abs_res, decreasing = TRUE, na.last = NA)[1:8]
top_pairs <- arrayInd(top_idx, dim(abs_res))
top_table <- data.frame(
  var1 = colnames(Xz)[top_pairs[,1]],
  var2 = colnames(Xz)[top_pairs[,2]],
  resid = round(resid_R[top_pairs], 3)
)
cat("\n--- Largest residual correlations ---\n")
print(top_table)

############################################################
# 7) Factor scores + a simple downstream use
############################################################

scores <- fa_oblimin$scores
colnames(scores) <- paste0("Factor", 1:m)

cat("\n--- Correlation of factor scores (estimated) ---\n")
print(round(cor(scores), 3))

# Example: clustering individuals in factor-score space (toy demo)
# (Often used in psychometrics / marketing segmentation)
km <- kmeans(scores, centers = 3, nstart = 20)
cat("\n--- K-means cluster sizes on factor scores ---\n")
print(km$size)

############################################################
# 8) (Optional) In simulation: compare estimated factor scores to true factors
# Note: factor scores are only determined up to rotation/sign in general.
############################################################

# Align estimated scores with true factors by maximal absolute correlation.
# This is only meaningful here because we simulated the data.
C <- cor(scores, F_latent)
cat("\n--- Cor(scores, true factors) (unmatched) ---\n")
print(round(C, 3))

# Greedy matching
match <- integer(m)
used <- rep(FALSE, k)
for (j in 1:m) {
  best <- which.max(ifelse(used, -Inf, abs(C[j, ])))
  match[j] <- best
  used[best] <- TRUE
}
C_matched <- sapply(1:m, \(j) C[j, match[j]])
cat("\n--- Approx alignment: matched correlations per factor score ---\n")
print(round(C_matched, 3))

############################################################
# Notes for students:
# - Compare varimax vs oblimin: interpretability vs allowing factor correlation
# - Inspect h2/u2: low h2 suggests variable not well explained by common factors
# - Check residual correlations: large residuals suggest misfit / too few factors
# - Parallel analysis is a common heuristic, not a proof
############################################################