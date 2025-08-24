# diversity_sampler_r/init.R
# R equivalent of __init__.py - package initialization

# Load all required packages
library(digest)
library(jsonlite)

# Source all modules
source("pool.R")
source("dpp.R")
source("metrics.R")
source("kernels.R")
source("selectors.R")

# Export main functions and classes
# Note: In R, we don't need explicit exports like Python's __all__
# All functions and classes are available after sourcing

cat("Diversity Sampler R package loaded successfully!\n")
cat("Available main functions:\n")
cat("- CandidatePool: Streaming candidate pool\n")
cat("- k_dpp_sample: k-DPP sampling\n")
cat("- select_diverse: Greedy diverse selection\n")
cat("- build_kernel: Kernel construction\n")
cat("- diversity_metrics: Evaluation metrics\n") 