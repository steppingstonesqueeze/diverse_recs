# diversity_sampler_r/metrics.R
# R equivalent of metrics.py

diversity_metrics <- function(X, idxs) {
  idxs <- as.integer(idxs)
  if (length(idxs) == 0) {
    return(list(k = 0))
  }
  
  Xs <- X[idxs, , drop = FALSE]
  
  # mean pairwise cosine similarity
  Xn <- Xs / pmax(sqrt(rowSums(Xs^2)), 1e-12)
  S <- Xn %*% t(Xn)
  n <- length(idxs)
  
  # Get upper triangle indices (excluding diagonal)
  iu <- upper.tri(S, diag = FALSE)
  mean_pairwise_cos <- if (sum(iu) > 0) mean(S[iu]) else 1.0
  
  # nearest-neighbor cosine distance
  S_diag <- S
  diag(S_diag) <- -Inf
  nn_sim <- apply(S_diag, 1, max)
  mean_nn_cos_dist <- mean(1.0 - nn_sim)
  
  return(list(
    k = n,
    mean_pairwise_cosine = mean_pairwise_cos,
    mean_nn_cosine_distance = mean_nn_cos_dist
  ))
} 