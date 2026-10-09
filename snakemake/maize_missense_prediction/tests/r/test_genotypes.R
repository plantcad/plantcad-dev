testthat::test_that("centering makes kernels invariant to allele reversal", {
  X <- matrix(c(0, 1, 2, 0, 2, 2, 1, 0, 1, 1, 0, 2), nrow = 4)
  w <- c(1, 0.4, 0.4)
  expected <- sweep(X, 2, colMeans(X), "-") %*% diag(w) %*% t(sweep(X, 2, colMeans(X), "-"))
  observed <- weighted_block(X, w)$kernel
  testthat::expect_equal(observed, expected)
  reversed <- X
  reversed[, 2] <- 2 - reversed[, 2]
  testthat::expect_equal(weighted_block(reversed, w)$kernel, observed)
  testthat::expect_true(min(eigen(observed, symmetric = TRUE)$values) > -1e-10)
  rescaled <- weighted_block(X, 0.031 * w)$kernel
  testthat::expect_equal(rescaled / mean(diag(rescaled)), observed / mean(diag(observed)),
                         tolerance = 1e-12)
})

testthat::test_that("mean imputation and monomorphic columns are handled", {
  X <- cbind(c(0, 2, NA, 1), rep(1, 4), rep(NA, 4))
  result <- weighted_block(X, c(1, 1, 1))
  testthat::expect_equal(result$missing, 5)
  testthat::expect_equal(result$monomorphic, 2)
  testthat::expect_equal(result$kernel, tcrossprod(c(-1, 1, 0, 0)))
})

testthat::test_that("GDS access respects IDs and chunk boundaries", {
  folder <- withr::local_tempdir()
  gds <- file.path(folder, "toy.gds")
  X <- matrix(c(0, 1, 2, 1, 2, 0, 0, 2, 1, 1, 0, 2), nrow = 4)
  SNPRelate::snpgdsCreateGeno(gds, X, sample.id = c("s1", "s2", "s3", "s4"),
                             snp.id = c(21L, 5L, 90L), snp.chromosome = c(1, 1, 1),
                             snp.position = c(10, 20, 30), snp.allele = rep("A/C", 3),
                             snpfirstdim = FALSE)
  weights <- file.path(folder, "weights.parquet")
  arrow::write_parquet(data.frame(snp_id = c("90", "21", "5"), weight = c(0.4, 1, 0.4)), weights)
  K1 <- build_kernel(gds, weights, file.path(folder, "one.rds"), file.path(folder, "one.json"), 1)
  K3 <- build_kernel(gds, weights, file.path(folder, "all.rds"), file.path(folder, "all.json"), 3)
  expected <- weighted_block(X, c(1, 0.4, 0.4))$kernel
  expected <- expected / mean(diag(expected))
  testthat::expect_equal(unname(K1), expected)
  testthat::expect_equal(K1, K3)
  metadata <- file.path(folder, "metadata.tsv.gz")
  extract_gds_metadata(gds, metadata, chunk_size = 1)
  testthat::expect_equal(read.delim(metadata)$snp_id, c(21L, 5L, 90L))
})
