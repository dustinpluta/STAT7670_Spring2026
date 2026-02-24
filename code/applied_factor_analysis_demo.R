############################################################
# Applied EFA workflow (graduate multivariate stats)
# Dataset: psych::bfi (Big Five Inventory-like items)
############################################################

set.seed(2026)

# Packages
pkgs <- c("psych", "GPArotation", "dplyr")
to_install <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(to_install) > 0) install.packages(to_install)
lapply(pkgs, require, character.only = TRUE)

############################################################
# 1) Load data + choose variables
############################################################

data(bfi, package = "psych")

# The bfi data includes 25 personality items + some demographics.
# We'll use the 25 items (columns 1:25).
items <- bfi[, 1:25]

# Quick look
cat("N rows:", nrow(items), " | N items:", ncol(items), "\n")

############################################################
# 2) Data screening / cleaning
############################################################

# Missingness
miss_prop <- colMeans(is.na(items))
cat("\n--- Missingness proportion (top 10) ---\n")
print(sort(round(miss_prop, 3), decreasing = TRUE)[1:10])

# Simple strategy for teaching: impute by median (robust) item-wise.
# (Alternatives: psych::impute, multiple imputation, FIML; discuss in lecture.)
items_imp <- items %>%
  mutate(across(everything(), \(x) {
    x[is.na(x)] <- median(x, na.rm = TRUE)
    x
  }))

# Check for near-zero variance items (not common here, but good habit)
vars <- apply(items_imp, 2, var)
cat("\n--- Min/Max item variance ---\n")
print(range(vars))

############################################################
# 3) Factorability checks: correlation, KMO, Bartlett
############################################################

R <- cor(items_imp)

cat("\n--- KMO measure of sampling adequacy ---\n")
print(psych::KMO(R))

cat("\n--- Bartlett test of sphericity (H0: R = I) ---\n")
print(psych::cortest.bartlett(R, n = nrow(items_imp)))

# Optional: determinant of correlation matrix (very small can indicate multicollinearity)
cat("\n--- det(R) ---\n")
print(det(R))

############################################################
# 4) Decide number of factors: scree + parallel analysis
############################################################

# Parallel analysis for FA (not PCA): set fa="fa"
cat("\n--- Parallel analysis (FA) ---\n")
psych::fa.parallel(items_imp, fm = "ml", fa = "fa", n.iter = 50,
                   main = "Parallel Analysis (FA) on bfi items")

# For a Big Five instrument, 5 factors is a natural candidate,
# but teach students to reconcile: theory + parallel analysis + interpretability.
m <- 5

############################################################
# 5) Fit EFA: ML extraction + oblique rotation
############################################################

# Oblique is usually appropriate: latent traits often correlate.
efa <- psych::fa(items_imp, nfactors = m, fm = "ml",
                 rotate = "oblimin", scores = "regression")

cat("\n=== EFA results (ML, oblimin) ===\n")
print(efa)

cat("\n--- Pattern loadings (cut = .30), sorted ---\n")
print(psych::print.psych(efa$loadings, cut = 0.30, sort = TRUE))

cat("\n--- Factor correlations (Phi) ---\n")
print(round(efa$Phi, 3))

############################################################
# 6) Interpretation workflow: identify item clusters per factor
############################################################

L <- as.matrix(efa$loadings)

# For each item, find the factor with largest absolute loading
primary_factor <- apply(abs(L), 1, which.max)
primary_loading <- L[cbind(1:nrow(L), primary_factor)]

interpret_table <- data.frame(
  item = rownames(L),
  primary_factor = primary_factor,
  primary_loading = round(primary_loading, 3),
  h2 = round(efa$communality, 3),
  u2 = round(efa$uniquenesses, 3)
) %>%
  arrange(primary_factor, desc(abs(primary_loading)))

cat("\n--- Item -> primary factor assignment (by max |loading|) ---\n")
print(interpret_table)

# Flag potential issues: low communality or weak primary loading
cat("\n--- Flags: |primary_loading| < .30 or h2 < .20 ---\n")
flags <- interpret_table %>%
  filter(abs(primary_loading) < 0.30 | h2 < 0.20)
print(flags)

############################################################
# 7) Model diagnostics: residual correlations / misfit
############################################################

resid_R <- efa$residual
abs_res <- abs(resid_R)
abs_res[lower.tri(abs_res, diag = TRUE)] <- NA

cat("\n--- Residual correlation summary ---\n")
cat("Max |residual|:", round(max(abs_res, na.rm = TRUE), 3), "\n")
cat("Mean |residual|:", round(mean(abs_res, na.rm = TRUE), 3), "\n")

# Show top residual pairs (possible cross-loadings / local dependence)
top_idx <- order(abs_res, decreasing = TRUE, na.last = NA)[1:10]
top_pairs <- arrayInd(top_idx, dim(abs_res))
top_table <- data.frame(
  item1 = colnames(items_imp)[top_pairs[,1]],
  item2 = colnames(items_imp)[top_pairs[,2]],
  resid = round(resid_R[top_pairs], 3)
)

cat("\n--- Top residual correlations (possible misfit) ---\n")
print(top_table)

############################################################
# 8) Factor scores: downstream analysis
############################################################

scores <- as.data.frame(efa$scores)
colnames(scores) <- paste0("F", 1:m)

cat("\n--- Correlation of factor scores ---\n")
print(round(cor(scores), 3))

# Example downstream use:
# Predict a demographic variable (e.g., gender) from factor scores
# NOTE: In bfi, gender is coded in the dataset (sex). We'll pull it.
# Some versions use "gender" or "sex"; check what's present.
demo_cols <- intersect(c("sex", "gender", "age", "education"), colnames(bfi))
cat("\n--- Available demographics in bfi ---\n")
print(demo_cols)

if ("sex" %in% colnames(bfi)) {
  y <- bfi$sex
  # Fit a simple model (illustrative only)
  df_model <- cbind(y = y, scores)
  df_model <- df_model[complete.cases(df_model), ]
  
  cat("\n--- Logistic regression of sex on factor scores (illustrative) ---\n")
  fit <- glm(y ~ ., data = df_model, family = binomial())
  print(summary(fit))
}

############################################################
# 9) Sensitivity analysis: compare varimax vs oblimin
############################################################

efa_varimax <- psych::fa(items_imp, nfactors = m, fm = "ml",
                         rotate = "varimax", scores = "regression")

cat("\n=== Compare rotations: varimax vs oblimin ===\n")
cat("\nVarimax loadings (cut=.30):\n")
print(psych::print.psych(efa_varimax$loadings, cut = 0.30, sort = TRUE))

cat("\nOblimin loadings (cut=.30):\n")
print(psych::print.psych(efa$loadings, cut = 0.30, sort = TRUE))

############################################################
# Teaching notes (talking points):
# - Why KMO/Bartlett: factorability & common variance plausibility
# - Why oblique: trait factors often correlated; varimax forces orthogonality
# - How to choose m: parallel analysis + theory + interpretability + residuals
# - How to read h2/u2: common vs unique variance (and why uniqueness matters)
# - Residual correlations: local dependence / missing cross-loadings / too few factors
# - Factor scores: convenient but not "observed"; discuss indeterminacy
############################################################