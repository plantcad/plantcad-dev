source("src/r/genotypes.R")
source("src/r/evaluation.R")
testthat::test_dir("tests/r", stop_on_failure = TRUE)
