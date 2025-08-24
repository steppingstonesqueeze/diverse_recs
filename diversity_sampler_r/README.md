# Diversity Sampler - R Version

This is the R equivalent of the Python diversity sampler codebase, implementing submodular maximization and Determinantal Point Processes (DPPs) for diverse subset selection.

## Overview

The codebase provides algorithms for selecting diverse subsets from large candidate pools, with applications in recommendation systems, data summarization, and active learning.

## Key Components

### Core Modules

- **pool.R**: Streaming candidate pool with mergeability (BOTTOMK and PRIORITY modes)
- **dpp.R**: k-DPP sampling implementation
- **metrics.R**: Diversity metrics calculation
- **kernels.R**: Kernel functions (cosine, RBF)
- **selectors.R**: Greedy selection algorithms (farthest-first, facility location, sum diversity)

### Main Functions

- `CandidatePool`: Streaming pool for managing candidates
- `k_dpp_sample`: k-DPP sampling from L-ensemble
- `select_diverse`: Greedy diverse subset selection
- `build_kernel`: Kernel matrix construction
- `diversity_metrics`: Evaluation metrics

## Installation

Required R packages:
```r
install.packages(c("digest", "jsonlite"))
```

## Usage

### Basic Example

```r
# Source the main module
source("diversity_sampler.R")

# Create candidate pool
pool <- CandidatePool$new(capacity = 1000, mode = PoolMode$BOTTOMK, seed = 42)

# Add candidates
for (i in 1:n) {
  pool$add(item_id = i, features = X[i, ], weight = 1.0)
}

# Get candidates
result <- pool$candidates()
Xcand <- result$X

# Select diverse subset
sel_idxs <- select_diverse(Xcand, k = 50, 
                          objective = SelectionObjective$FACILITY_LOCATION)

# Evaluate diversity
metrics <- diversity_metrics(Xcand, sel_idxs)
```

### Running Examples

```r
# Example 1: Basic demo
source("example_usage_v1.R")

# Example 2: Full demo with CSV output
source("example_usage_v2.R")
```

## Algorithms

### Selection Objectives

1. **FARTHEST_FIRST**: Gonzalez farthest-first traversal (max-min)
2. **FACILITY_LOCATION**: Greedy facility location (coverage via similarity)
3. **SUM_DIVERSITY**: Minimize pairwise inner products

### Pool Modes

1. **BOTTOMK**: Uniform sampling without replacement via k smallest hash
2. **PRIORITY**: Weighted PPSWOR via priority sampling

## Data Files

The examples use the following CSV files (same as Python version):
- `candidates_1000.csv`: 1000 candidate points
- `greedy50.csv`: 50 points selected by greedy facility location
- `kdpp50.csv`: 50 points selected by k-DPP

## Performance

The R implementation maintains the same algorithmic complexity as the Python version:
- Greedy algorithms: O(n k d)
- k-DPP sampling: O(n³) for eigendecomposition + O(k²) for sampling

## Differences from Python Version

- Uses R's `setRefClass` for object-oriented programming
- Matrix operations use R's native matrix algebra
- Random number generation uses R's `set.seed()` and `runif()`
- File I/O uses R's `write.csv()` function 