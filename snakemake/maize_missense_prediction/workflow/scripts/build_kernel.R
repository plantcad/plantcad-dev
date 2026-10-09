source("src/r/genotypes.R")
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 6)
build_kernel(args[1], args[2], args[3], args[4], as.integer(args[5]), as.logical(args[6]))
