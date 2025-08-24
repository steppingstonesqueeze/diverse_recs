# diversity_sampler_r/pool.R
# R equivalent of pool.py

library(digest)
library(jsonlite)

# Hash functions equivalent to Python's stable_hash_any
hash64_to_unit <- function(x) {
  # Map a 64-bit integer to (0,1)
  return((x %% (2^64)) / (2^64))
}

stable_hash_bytes <- function(b) {
  # 64-bit stable hash using digest
  h <- digest(b, algo = "sha256", raw = TRUE)
  # Take first 8 bytes and convert to integer
  # Convert raw bytes to integer using bitwise operations
  result <- 0
  for (i in 1:min(8, length(h))) {
    result <- result + as.integer(h[i]) * (256^(i-1))
  }
  return(result)
}

stable_hash_any <- function(obj) {
  if (is.raw(obj)) {
    return(stable_hash_bytes(obj))
  }
  if (is.character(obj)) {
    return(stable_hash_bytes(charToRaw(obj)))
  }
  if (is.numeric(obj) && length(obj) == 1 && obj == as.integer(obj)) {
    return(obj %% (2^64))
  }
  # Fallback: JSON-serialize
  json_str <- toJSON(obj, auto_unbox = TRUE, digits = 20)
  return(stable_hash_bytes(charToRaw(json_str)))
}

# Pool modes
PoolMode <- list(
  BOTTOMK = "bottomk",      # uniform without replacement via k smallest hash
  PRIORITY = "priority"     # weighted PPSWOR via priority sampling (keep largest priority)
)

# Heap item structure
HeapItem <- setRefClass("HeapItem",
  fields = list(
    key = "numeric",
    item_id = "ANY",
    idx = "integer"
  ),
  methods = list(
    initialize = function(key, item_id, idx) {
      .self$key <- key
      .self$item_id <- item_id
      .self$idx <- idx
    }
  )
)

# CandidatePool class
CandidatePool <- setRefClass("CandidatePool",
  fields = list(
    capacity = "integer",
    mode = "character",
    heap = "list",
    items = "list",
    rng = "ANY",
    next_idx = "integer"
  ),
  methods = list(
    initialize = function(capacity, mode = PoolMode$BOTTOMK, seed = NULL) {
      if (capacity <= 0) stop("capacity must be positive")
      if (!(mode %in% c(PoolMode$BOTTOMK, PoolMode$PRIORITY))) {
        stop("mode must be either BOTTOMK or PRIORITY")
      }
      .self$capacity <- as.integer(capacity)
      .self$mode <- mode
      .self$heap <- list()
      .self$items <- list()
      # item_id -> (features, weight, timestamp, metadata, priority_key)
      .self$rng <- if (!is.null(seed)) set.seed(seed) else NULL
      .self$next_idx <- 1L
    },
    
    get_length = function() {
      return(length(.self$items))
    },
    
    heap_key = function(priority) {
      return(priority)
    },
    
    maybe_insert = function(item_id, key_value, features, weight, ts, metadata) {
      # For BOTTOMK: keep k smallest key_value
      # For PRIORITY: keep k largest key_value
      if (.self$mode == PoolMode$BOTTOMK) {
        # If already present and we got a smaller key, keep smallest
        if (!is.null(.self$items[[as.character(item_id)]])) {
          prev <- .self$items[[as.character(item_id)]][[5]]
          if (key_value < prev) {
            .self$items[[as.character(item_id)]] <- list(features, weight, ts, metadata, key_value)
          } else {
            # Update features/weight/ts/metadata even if key same
            .self$items[[as.character(item_id)]] <- list(features, weight, ts, metadata, prev)
          }
          return()
        }
        
        if (length(.self$heap) < .self$capacity) {
          .self$heap[[length(.self$heap) + 1]] <- HeapItem$new(key_value, item_id, .self$next_idx)
          .self$next_idx <- .self$next_idx + 1L
          .self$items[[as.character(item_id)]] <- list(features, weight, ts, metadata, key_value)
        } else {
          # Find worst key (max)
          worst_key <- max(sapply(.self$heap, function(x) x$key))
          if (key_value < worst_key) {
            # Find index of worst
            worst_idx <- which.max(sapply(.self$heap, function(x) x$key))
            removed <- .self$heap[[worst_idx]]
            .self$heap <- .self$heap[-worst_idx]
            # Delete from items
            if (!is.null(.self$items[[as.character(removed$item_id)]])) {
              .self$items[[as.character(removed$item_id)]] <- NULL
            }
            # Insert new
            .self$heap[[length(.self$heap) + 1]] <- HeapItem$new(key_value, item_id, .self$next_idx)
            .self$next_idx <- .self$next_idx + 1L
            .self$items[[as.character(item_id)]] <- list(features, weight, ts, metadata, key_value)
          }
        }
      } else {
        # PRIORITY: keep k largest key_value
        if (!is.null(.self$items[[as.character(item_id)]])) {
          prev <- .self$items[[as.character(item_id)]][[5]]
          if (key_value > prev) {
            .self$items[[as.character(item_id)]] <- list(features, weight, ts, metadata, key_value)
          } else {
            .self$items[[as.character(item_id)]] <- list(features, weight, ts, metadata, prev)
          }
          return()
        }
        
        if (length(.self$heap) < .self$capacity) {
          .self$heap[[length(.self$heap) + 1]] <- HeapItem$new(-key_value, item_id, .self$next_idx)
          .self$next_idx <- .self$next_idx + 1L
          .self$items[[as.character(item_id)]] <- list(features, weight, ts, metadata, key_value)
        } else {
          # Worst is smallest priority => largest negative key
          smallest_priority <- -min(sapply(.self$heap, function(x) x$key))
          if (key_value > smallest_priority) {
            worst_idx <- which.min(sapply(.self$heap, function(x) x$key))
            removed <- .self$heap[[worst_idx]]
            .self$heap <- .self$heap[-worst_idx]
            if (!is.null(.self$items[[as.character(removed$item_id)]])) {
              .self$items[[as.character(removed$item_id)]] <- NULL
            }
            .self$heap[[length(.self$heap) + 1]] <- HeapItem$new(-key_value, item_id, .self$next_idx)
            .self$next_idx <- .self$next_idx + 1L
            .self$items[[as.character(item_id)]] <- list(features, weight, ts, metadata, key_value)
          }
        }
      }
    },
    
    add = function(item_id, features, weight = 1.0, ts = NULL, metadata = NULL) {
      if (is.null(ts)) ts <- as.numeric(Sys.time())
      features <- as.numeric(features)
      if (.self$mode == PoolMode$BOTTOMK) {
        h <- stable_hash_any(item_id)
        key <- hash64_to_unit(h)
      } else {
        if (weight <= 0) weight <- 1e-9
        u <- runif(1)
        if (u <= 0.0) u <- .Machine$double.eps
        key <- u^(1.0 / weight)  # keep k largest
      }
      .self$maybe_insert(item_id, key, features, weight, ts, metadata)
    },
    
    merge = function(other) {
      if (.self$mode != other$mode) stop("modes must match")
      # Just add other's items through maybe_insert with their stored key
      for (item_id in names(other$items)) {
        item_data <- other$items[[item_id]]
        .self$maybe_insert(item_id, item_data[[5]], item_data[[1]], item_data[[2]], item_data[[3]], item_data[[4]])
      }
    },
    
    candidates = function() {
      # Return (ids, X, weights, timestamps, metadata) with consistent order
      ids <- names(.self$items)
      if (length(ids) == 0) {
        return(list(ids = character(0), X = matrix(numeric(0), 0, 0), W = numeric(0), TS = numeric(0), META = list()))
      }
      
      X <- do.call(rbind, lapply(.self$items, function(x) x[[1]]))
      W <- sapply(.self$items, function(x) x[[2]])
      TS <- sapply(.self$items, function(x) x[[3]])
      META <- lapply(.self$items, function(x) x[[4]])
      
      return(list(ids = ids, X = X, W = W, TS = TS, META = META))
    }
  )
) 