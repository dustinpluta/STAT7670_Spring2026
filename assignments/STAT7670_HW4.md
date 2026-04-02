# STAT — Classification Methods Assignment

**Topic:** LDA, Logistic Regression, SVMs, Decision Trees, Random Forests  
**Format:** Computational (R)  
**Due:** 2026-04-19
**Assigned:** 2026-04-02

---

## Instructions

- Use **R** for all computations.
- Submit:
  - R code
  - requested plots and tables
  - clear written interpretations
- Set a random seed where appropriate.
- Clearly label all outputs.
- Write concise but complete answers.

---

# Problem 1 — Linear Discriminant Analysis (LDA) on `iris`

## Tasks

### (a) Exploratory analysis
- Table of class counts
- Pairs plot colored by species
- Table of class means

**Question:** Which variables separate classes best?

---

### (b) Fit LDA
Report:
- prior probabilities
- group means
- discriminant coefficients

**Question:** Interpret LD1.

---

### (c) Classification performance
- 70/30 split
- confusion matrix
- test accuracy

**Question:** Which classes are hardest to separate?

---

### (d) Plot
- LD1 vs LD2 colored by species

**Question:** What geometry do you observe?

---

### (e) Reduced model
- Use only petal variables

**Question:** Compare performance.

---

# Problem 2 — Logistic Regression on `iris` (Binary)

Define:
- 1 = virginica, 0 = otherwise

---

### (a) Visualization
- pairs plot
- class proportions

---

### (b) Fit model
Report:
- coefficients
- p-values
- odds ratios

Interpret two coefficients.

---

### (c) Classification
- train/test split
- confusion matrix
- sensitivity/specificity

---

### (d) Probability surface
- 2D grid plot (petal variables)
- include 0.5 boundary

---

### (e) Threshold comparison
Table with:
- sensitivity
- specificity
- accuracy

---

# Problem 3 — SVM on Synthetic Data

Generate nonlinear data.

---

### (a) Plot data
Explain nonlinearity.

---

### (b) Linear SVM
- accuracy
- confusion matrix
- support vectors

---

### (c) RBF SVM
- tune parameters
- report accuracy

---

### (d) Comparison table

---

### (e) Plot boundary
Include support vectors.

---

# Problem 4 — Decision Trees

Use `iris`.

---

### (a) Fit tree
- print cp table
- plot tree

---

### (b) Performance
- train/test accuracy

---

### (c) Pruning
- prune tree
- compare performance

---

### (d) Probabilities
Display first 10 predictions.

---

### (e) Interpretation
Short paragraph.

---

# Problem 5 — Random Forests vs Trees

Synthetic nonlinear data.

---

### (a) Plot data

---

### (b) Fit tree
- accuracy

---

### (c) Fit random forest
- OOB error
- test accuracy

---

### (d) Variable importance
Interpret results.

---

### (e) Comparison table

---

### (f) Simulation study
- 25 repetitions
- boxplot
- summary stats

---

# Optional Problem 6 — Method Comparison

Compare at least 3 classifiers.

---

# Grading

- Code correctness: 30%
- Plots/tables: 25%
- Interpretation: 25%
- Clarity: 20%

---

# Submission

Submit:
- R script or R Markdown file
- PDF or HTML output
