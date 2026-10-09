source("src/r/evaluation.R")
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 12)
evaluate(args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9],
          strsplit(args[10], ",", fixed = TRUE)[[1]], as.integer(args[11]), as.numeric(args[12]))
