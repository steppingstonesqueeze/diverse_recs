# diversity_sampler_r/example_usage_v2.R
# R equivalent of example_usage_v2.py

# Source the main diversity sampler
source("diversity_sampler.R")

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
}

demo <- function() {
  set.seed(0)
  n <- 2000
  d <- 64
  X <- matrix(rnorm(n * d), n, d)
  X <- X / sqrt(rowSums(X^2))  # normalize for cosine
  
  # 1) Candidate pool (uniform bottom-k by hash) -> 1000 points
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
  cat("Candidates:", length(ids), "\n")
  
  # Save the 1000 candidate points
  save_csv("candidates_1000.csv", ids, Xcand)
  
  # 2A) Greedy facility-location (k=50)
  k <- 50
  greedy_idx <- select_diverse(
    Xcand, k = k,
    objective = SelectionObjective$FACILITY_LOCATION,
    metric = "cosine", seed = 0
  )
  cat("Greedy facility-location metrics:", "\n")
  print(diversity_metrics(Xcand, greedy_idx))
  greedy_ids <- ids[greedy_idx]
  save_csv("greedy50.csv", greedy_ids, Xcand[greedy_idx, ])
  
  # 2B) k-DPP (k=50) on a better-conditioned kernel + fallback fill
  # Tip: include (1-alpha)*I and a touch more jitter to stabilize rank.
  L <- build_kernel(
    Xcand,
    quality = NULL,
    kind = "cosine",   # try "rbf" if you prefer
    alpha = 0.9,
    jitter = 1e-5
  )
  dpp_idx <- k_dpp_sample(L, k = k, seed = 0)
  
  # Fallback: if numerics return < k, fill with farthest-first on the remainder
  if (length(dpp_idx) < k) {
    remaining <- setdiff(1:length(ids), dpp_idx)
    fill_idx_local <- select_diverse(
      Xcand[remaining, ], k = k - length(dpp_idx),
      objective = SelectionObjective$FARTHEST_FIRST,
      metric = "cosine", seed = 1
    )
    dpp_idx <- c(dpp_idx, remaining[fill_idx_local])
  }
  
  cat("k-DPP metrics:", "\n")
  print(diversity_metrics(Xcand, dpp_idx))
  dpp_ids <- ids[dpp_idx]
  save_csv("kdpp50.csv", dpp_ids, Xcand[dpp_idx, ])
}

# Run the demo
demo() 