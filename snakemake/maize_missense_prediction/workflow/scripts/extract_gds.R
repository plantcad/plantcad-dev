source("src/r/genotypes.R")
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2)
extract_gds_metadata(args[1], args[2])
