# diversity_sampler_r/example_usage_v1.R
# R equivalent of example_usage_v1.py

# Source the main diversity sampler
source("diversity_sampler.R")

demo <- function() {
  # Simulate a stream
  set.seed(0)
  n <- 2000
  d <- 64
  X <- matrix(rnorm(n * d), n, d)
  X <- X / sqrt(rowSums(X^2))  # normalize for cosine
  # quality scores (optional)
  q <- pmin(pmax(rlnorm(n, meanlog = 0.0, sdlog = 1.0), 0), 10)
  
  # 1) Build candidate pool (uniform bottom-k)
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
  
  # 2A) Greedy selection (facility location)
  k <- 50
  sel_idxs <- select_diverse(Xcand, k = k, objective = SelectionObjective$FACILITY_LOCATION, metric = "cosine", seed = 0)
  cat("Greedy facility-location metrics:", "\n")
  print(diversity_metrics(Xcand, sel_idxs))
  
  # 2B) k-DPP sampling on same candidates
  k <- 50
  L <- build_kernel(Xcand, quality = NULL, kind = "cosine", alpha = 0.9, jitter = 1e-6)
  Y <- k_dpp_sample(L, k = k, seed = 0)
  cat("k-DPP metrics:", "\n")
  print(diversity_metrics(Xcand, Y))
}

# Run the demo
demo() 