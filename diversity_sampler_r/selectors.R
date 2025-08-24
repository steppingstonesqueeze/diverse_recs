# diversity_sampler_r/selectors.R
# R equivalent of selectors.py

# Selection objectives
SelectionObjective <- list(
  FARTHEST_FIRST = "farthest_first",           # max-min (Gonzalez)
  FACILITY_LOCATION = "facility_location",     # coverage via similarity
  SUM_DIVERSITY = "sum_diversity"              # minimize pairwise inner products
)

l2_normalize <- function(X, eps = 1e-12) {
  n <- sqrt(rowSums(X^2))
  n <- pmax(n, eps)
  return(X / n)
}

farthest_first <- function(X, k, metric = "euclidean", seed = NULL) {
  # Gonzalez farthest-first traversal. Returns indices of selected points.
  # metric: 'euclidean' or 'cosine' (assumes X normalized if cosine).
  # Complexity: O(n k d).
  if (!is.null(seed)) set.seed(seed)
  
  n <- nrow(X)
  if (n == 0 || k <= 0) return(integer(0))
  if (k >= n) return(1:n)
  
  if (metric == "cosine") {
    Xn <- l2_normalize(X)
    # cosine distance = 1 - dot
    dists <- rep(Inf, n)
    start <- sample(1:n, 1)
    selected <- start
    dists <- pmin(dists, 1.0 - (Xn %*% Xn[start, ]))
    
    for (iter in 2:k) {
      i <- which.max(dists)
      selected <- c(selected, i)
      dists <- pmin(dists, 1.0 - (Xn %*% Xn[i, ]))
    }
    return(selected)
  } else {
    dists <- rep(Inf, n)
    start <- sample(1:n, 1)
    selected <- start
    diff <- X - matrix(X[start, ], n, ncol(X), byrow = TRUE)
    dists <- pmin(dists, rowSums(diff^2))
    
    for (iter in 2:k) {
      i <- which.max(dists)
      selected <- c(selected, i)
      diff <- X - matrix(X[i, ], n, ncol(X), byrow = TRUE)
      dists <- pmin(dists, rowSums(diff^2))
    }
    return(selected)
  }
}

facility_location_greedy <- function(X, k, metric = "cosine") {
  # Greedy maximize sum_i max_{s in S} sim(i, s).
  # For 'cosine', uses dot products on l2-normalized X.
  # Complexity: O(n k d).
  n <- nrow(X)
  if (n == 0 || k <= 0) return(integer(0))
  if (k >= n) return(1:n)
  
  if (metric == "cosine") {
    Xn <- l2_normalize(X)
    best <- rep(-Inf, n)
    selected <- integer(0)
    
    for (iter in 1:k) {
      # compute gain of adding each candidate j: sum_i max(0, Xn[i]·Xn[j] - best[i])
      sims <- Xn %*% t(Xn)   # we can avoid full matrix by caching; for clarity keep this
      gains <- pmax(0.0, sims - matrix(best, n, n))
      # Ensure gains is a matrix
      if (!is.matrix(gains)) gains <- matrix(gains, nrow = n, ncol = n)
      # Sum gains per j
      gsum <- colSums(gains, na.rm = TRUE)
      # Avoid already selected items
      gsum[selected] <- -Inf
      j <- which.max(gsum)
      # Safety check
      if (length(j) == 0 || j > n) j <- 1
      selected <- c(selected, j)
      best <- pmax(best, sims[, j])
    }
    # Deduplicate indices (in case numerical ties)
    return(unique(selected)[1:min(length(unique(selected)), k)])
  } else {
    # Euclidean similarity: use negative squared distance as similarity for facility-location
    best <- rep(-Inf, n)
    selected <- integer(0)
    
    for (iter in 1:k) {
      # compute similarities -||x - x_j||^2
      # sim_ij = -||x_i||^2 - ||x_j||^2 + 2 x_i·x_j
      norms <- rowSums(X^2)
      dots <- X %*% t(X)
      sims <- -(outer(norms, norms, "+") - 2 * dots)
      gains <- pmax(0.0, sims - matrix(best, n, n))
      # Ensure gains is a matrix
      if (!is.matrix(gains)) gains <- matrix(gains, nrow = n, ncol = n)
      # Sum gains per j
      gsum <- colSums(gains, na.rm = TRUE)
      # Avoid already selected items
      gsum[selected] <- -Inf
      j <- which.max(gsum)
      # Safety check
      if (length(j) == 0 || j > n) j <- 1
      selected <- c(selected, j)
      best <- pmax(best, sims[, j])
    }
    return(unique(selected)[1:min(length(unique(selected)), k)])
  }
}

sum_diversity_greedy <- function(X, k, metric = "cosine") {
  # Greedy minimize sum of pairwise inner products (or maximize negative of that).
  # For unit-normalized X, minimizing sum dot equals minimizing ||sum(X_S)||^2.
  # Strategy: start with the vector of smallest correlation to the mean, then iteratively pick argmin x·s where s=sum of selected vectors.
  n <- nrow(X)
  if (n == 0 || k <= 0) return(integer(0))
  if (k >= n) return(1:n)
  
  if (metric == "cosine") {
    Xn <- l2_normalize(X)
    # start with the point most opposite to dataset mean
    mu <- colMeans(Xn)
    scores <- Xn %*% mu
    selected <- which.min(scores)
    s <- Xn[selected, ]
    
    for (iter in 2:k) {
      dots <- Xn %*% s
      # choose minimal dot (most orthogonal/opposite to sum)
      # avoid duplicates by setting already selected to +inf
      dots[selected] <- Inf
      j <- which.min(dots)
      selected <- c(selected, j)
      s <- s + Xn[j, ]
    }
    return(selected)
  } else {
    # For euclidean, we can use cosine on normalized; distance-only version falls back to farthest-first
    return(farthest_first(X, k, metric = "euclidean"))
  }
}

select_diverse <- function(X, k, objective = SelectionObjective$FARTHEST_FIRST, metric = "cosine", seed = NULL) {
  if (objective == SelectionObjective$FARTHEST_FIRST) {
    return(farthest_first(X, k, metric = metric, seed = seed))
  } else if (objective == SelectionObjective$FACILITY_LOCATION) {
    return(facility_location_greedy(X, k, metric = metric))
  } else if (objective == SelectionObjective$SUM_DIVERSITY) {
    return(sum_diversity_greedy(X, k, metric = metric))
  } else {
    stop(paste("Unknown objective:", objective))
  }
} 