# diversity_sampler_r/test.R
# Test script to verify the R implementation

# Source the main module
source("diversity_sampler.R")

cat("Testing Diversity Sampler R implementation...\n\n")

# Helper function to save CSV files
save_csv <- function(path, ids, X) {
  # Create directory if it doesn't exist
  dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
  
  # Create data frame with id and features
  df <- data.frame(id = ids)
  for (i in 1:ncol(X)) {
    df[[paste0("f", i-1)]] <- X[, i]
  }
  
  # Write CSV
  write.csv(df, path, row.names = FALSE)
  cat("Saved:", path, "\n")
}

# Test 1: Basic functionality with data saving
cat("Test 1: Basic functionality and data saving\n")
set.seed(42)
n <- 2000  # Use same size as original examples
d <- 64    # Use same dimension as original examples
X <- matrix(rnorm(n * d), n, d)
X <- X / sqrt(rowSums(X^2))  # normalize

# Test CandidatePool (same as original examples)
pool <- CandidatePool$new(capacity = 1000, mode = PoolMode$BOTTOMK, seed = 42)
for (i in 1:n) {
  pool$add(item_id = i, features = X[i, ], weight = 1.0)
}
result <- pool$candidates()
ids <- result$ids
Xcand <- result$X
W <- result$W
TS <- result$TS
META <- result$META
cat("Pool size:", length(ids), "\n")

# Save the candidate set (equivalent to candidates_1000.csv)
save_csv("candidates_1000.csv", ids, Xcand)

# Test selection algorithms
k <- 50  # Use same k as original examples
sel_farthest <- select_diverse(Xcand, k = k, objective = SelectionObjective$FARTHEST_FIRST, seed = 0)
sel_facility <- select_diverse(Xcand, k = k, objective = SelectionObjective$FACILITY_LOCATION, seed = 0)
sel_sum <- select_diverse(Xcand, k = k, objective = SelectionObjective$SUM_DIVERSITY, seed = 0)

cat("Selection sizes:", length(sel_farthest), length(sel_facility), length(sel_sum), "\n")

# Save greedy facility location results (equivalent to greedy50.csv)
greedy_ids <- ids[sel_facility]
save_csv("greedy50.csv", greedy_ids, Xcand[sel_facility, ])

# Test metrics
metrics_farthest <- diversity_metrics(Xcand, sel_farthest)
metrics_facility <- diversity_metrics(Xcand, sel_facility)
metrics_sum <- diversity_metrics(Xcand, sel_sum)

cat("Metrics - Farthest First:", metrics_farthest$mean_pairwise_cosine, "\n")
cat("Metrics - Facility Location:", metrics_facility$mean_pairwise_cosine, "\n")
cat("Metrics - Sum Diversity:", metrics_sum$mean_pairwise_cosine, "\n")

# Test 2: Kernel functions
cat("\nTest 2: Kernel functions\n")
S_cosine <- cosine_similarity(Xcand)
S_rbf <- rbf_kernel(Xcand)
L <- build_kernel(Xcand, kind = "cosine", alpha = 0.9, jitter = 1e-5)  # Same as original

cat("Cosine similarity matrix size:", dim(S_cosine), "\n")
cat("RBF kernel matrix size:", dim(S_rbf), "\n")
cat("L-ensemble matrix size:", dim(L), "\n")

# Test 3: k-DPP sampling with data saving
cat("\nTest 3: k-DPP sampling and data saving\n")
dpp_idx <- k_dpp_sample(L, k = k, seed = 0)  # Same seed as original
cat("DPP sample size:", length(dpp_idx), "\n")

# Fallback: if numerics return < k, fill with farthest-first on the remainder
if (length(dpp_idx) < k) {
  remaining <- setdiff(1:length(ids), dpp_idx)
  fill_idx_local <- select_diverse(
    Xcand[remaining, ], k = k - length(dpp_idx),
    objective = SelectionObjective$FARTHEST_FIRST,
    metric = "cosine", seed = 1
  )
  dpp_idx <- c(dpp_idx, remaining[fill_idx_local])
  cat("DPP sample size after fallback:", length(dpp_idx), "\n")
}

if (length(dpp_idx) > 0) {
  dpp_metrics <- diversity_metrics(Xcand, dpp_idx)
  cat("DPP metrics:", dpp_metrics$mean_pairwise_cosine, "\n")
  
  # Save k-DPP results (equivalent to kdpp50.csv)
  dpp_ids <- ids[dpp_idx]
  save_csv("kdpp50.csv", dpp_ids, Xcand[dpp_idx, ])
}

cat("\nAll tests completed successfully!\n")
cat("Generated files:\n")
cat("- candidates_1000.csv: Candidate pool\n")
cat("- greedy50.csv: Greedy facility location selection\n")
cat("- kdpp50.csv: k-DPP selection\n") 