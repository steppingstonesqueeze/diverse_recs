# diversity_sampler_r/dpp.R
# R equivalent of dpp.py

select_k_eigenvectors <- function(lambda, k, rng) {
  n <- length(lambda)
  E <- matrix(0, k + 1, n)
  E[1, ] <- 1.0
  
  for (i in 1:n) {
    li <- lambda[i]
    jmax <- min(i, k)
    for (j in jmax:1) {
      prev <- if (i > 1) E[j, i - 1] else 0.0
      base <- if (i > 1 && j > 1) E[j - 1, i - 1] else (if (j == 1) 1.0 else 0.0)
      E[j, i] <- prev + li * base
    }
  }
  
  S <- integer(0)
  j <- k
  for (i in n:1) {
    if (j == 0) break
    denom <- E[j, i]
    if (denom <= 0) next
    base <- if (i > 1 && j > 1) E[j - 1, i - 1] else (if (j == 1) 1.0 else 0.0)
    prob <- (lambda[i] * base) / denom
    if (runif(1) < prob) {
      S <- c(S, i)
      j <- j - 1
    }
  }
  return(rev(S))
}

k_dpp_sample <- function(L, k, eps = 1e-12, seed = NULL) {
  # Sample a size-k subset from an L-ensemble DPP with PSD kernel L (n x n)
  n <- nrow(L)
  if (k <= 0) return(integer(0))
  if (k >= n) return(1:n)
  
  if (!is.null(seed)) set.seed(seed)
  
  # Eigenvalue decomposition
  eig <- eigen(L, symmetric = TRUE)
  vals <- pmax(eig$values, 0.0)
  vecs <- eig$vectors
  
  # 1) pick exactly k eigenvectors
  S_idx <- select_k_eigenvectors(vals, k, NULL)
  if (length(S_idx) < k) {
    S_idx <- order(vals, decreasing = TRUE)[1:k]
  }
  
  V <- vecs[, S_idx, drop = FALSE]  # (n x r) with r == k
  
  # 2) sequentially pick items
  Y <- integer(0)
  for (iter in 1:k) {
    # row norms squared -> selection probabilities over items
    P_rows <- rowSums(V * V)
    sP <- sum(P_rows)
    if (sP <= eps) break
    i <- sample(n, 1, prob = P_rows / sP)
    Y <- c(Y, i)
    
    # pick a column j with prob proportional to V[i, j]^2
    col_weights <- V[i, ]^2
    sC <- sum(col_weights)
    if (sC <= eps) break
    j <- sample(ncol(V), 1, prob = col_weights / sC)
    
    # v = V[, j] / V[i, j]
    denom <- V[i, j]
    if (abs(denom) <= eps) break
    v <- V[, j] / denom  # shape (n,)
    
    # Update: V <- V - v %*% t(V[i, ])
    V <- V - outer(v, V[i, ])  # (n x r) - (n x 1) %*% (1 x r)
    
    # Drop column j and re-orthonormalize
    if (ncol(V) <= 1) break
    V <- V[, -j, drop = FALSE]
    # QR decomposition for orthonormalization
    qr_result <- qr(V)
    V <- qr.Q(qr_result)
  }
  
  # Deduplicate in case of numerical repeats
  return(unique(Y)[1:min(length(unique(Y)), k)])
} 