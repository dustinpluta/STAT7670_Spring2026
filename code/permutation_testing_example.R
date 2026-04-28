# ============================================================
# Permutation Test: Difference in Means
# ============================================================

set.seed(123)

# Simulated data
group_A <- c(82, 85, 88, 90, 91, 87, 84, 86)
group_B <- c(78, 80, 79, 77, 81, 76, 82, 79)

# Observed statistic: mean difference
T_obs <- mean(group_A) - mean(group_B)
T_obs

# Combine data
scores <- c(group_A, group_B)
labels <- c(rep("A", length(group_A)),
            rep("B", length(group_B)))

# Number of permutations
B <- 5000
perm_stats <- numeric(B)

# Permutation loop
for (b in 1:B) {
  
  perm_labels <- sample(labels)
  
  A_vals <- scores[perm_labels == "A"]
  B_vals <- scores[perm_labels == "B"]
  
  perm_stats[b] <- mean(A_vals) - mean(B_vals)
}

# Two-sided p-value
p_val <- (sum(abs(perm_stats) >= abs(T_obs)) + 1) / (B + 1)

cat("Observed mean difference:", T_obs, "\n")
cat("Permutation p-value:", p_val, "\n")

# ------------------------------------------------------------
# Plot permutation distribution
# ------------------------------------------------------------

hist(perm_stats,
     breaks = 30,
     main = "Permutation Distribution",
     xlab = "Difference in Means",
     col = "lightgray",
     border = "white")

abline(v = T_obs, col = "red", lwd = 2)
abline(v = -T_obs, col = "red", lwd = 2)

legend("topright",
       legend = c("Observed statistic"),
       col = "red",
       lwd = 2)