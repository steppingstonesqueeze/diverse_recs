# diversity_sampler_r/diversity_sampler.R
# R equivalent of diversity_sampler.py - main file combining all functionality

# Source all the modules
source("pool.R")
source("dpp.R")
source("metrics.R")
source("kernels.R")
source("selectors.R")

# Additional functions that were in the main Python file
l2_normalize <- function(X, eps = 1e-12) {
  n <- sqrt(rowSums(X^2))
  n <- pmax(n, eps)
  return(X / n)
}

cosine_similarity <- function(X) {
  Xn <- l2_normalize(X)
  return(Xn %*% t(Xn))
}

rbf_kernel <- function(X, gamma = NULL) {
  # gamma default = 1 / median(pairwise squared distance)
  n <- nrow(X)
  norms <- rowSums(X^2)
  D2 <- outer(norms, norms, "+") - 2 * (X %*% t(X))
  
  if (is.null(gamma)) {
    # median of upper triangle distances (avoid zeros)
    iu <- upper.tri(D2, diag = FALSE)
    med <- if (sum(iu) > 0) median(D2[iu]) else 1.0
    if (med <= 0) med <- 1.0
    gamma <- 1.0 / med
  }
  
  K <- exp(-gamma * pmax(D2, 0.0))
  return(K)
}

build_kernel <- function(X, quality = NULL, kind = "cosine", alpha = 1.0, jitter = 1e-6) {
  if (kind == "cosine") {
    S <- cosine_similarity(X)
  } else if (kind == "rbf") {
    S <- rbf_kernel(X, gamma = NULL)
  } else {
    stop("Unknown kernel kind")
  }
  
  if (!(0.0 <= alpha && alpha <= 1.0)) {
    stop("alpha must be in [0,1]")
  }
  
  S <- alpha * S + (1.0 - alpha) * diag(nrow(S))
  
  if (!is.null(quality)) {
    q <- as.numeric(quality)
    q <- pmax(q, 0.0)
    D <- diag(q)
    L <- D %*% S %*% D
  } else {
    L <- S
  }
  
  if (jitter && jitter > 0) {
    L <- L + jitter * diag(nrow(L))
  }
  
  return(L)
} 