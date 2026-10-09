testthat::test_that("ID normalization rejects collisions", {
  testthat::expect_equal(normalize_ids(c("NAM_H.F1E1", "Ames_H.A1")), c("F1E1", "A1"))
  testthat::expect_error(normalize_ids(c("NAM_H.A", "A")))
})

testthat::test_that("CP and LOFO leave out the intended samples", {
  ids <- c("F1E1", "F1E2", "F2E1", "F2E2", "A1", "A2")
  cp <- validation_folds(ids, c(rep("NAM_H", 4), rep("Ames_H", 2)), "CP")
  testthat::expect_equal(cp$NAM_H, 1:4)
  testthat::expect_equal(cp$Ames_H, 5:6)
  lofo <- validation_folds(ids[1:4], rep("NAM_H", 4), "LOFO")
  testthat::expect_equal(lofo$F1, 1:2)
  testthat::expect_equal(lofo$F2, 3:4)
})

testthat::test_that("held-out outcomes never enter fitting", {
  y <- c(1, 2, 3, 4, 100, 200)
  X <- matrix(1, 6, 1)
  K <- matrix(0.1, 6, 6) + diag(6)
  fake_fit <- function(y, X, GRM, maxit, tol, ncores, verbose) {
    testthat::expect_equal(y, 1:4)
    testthat::expect_equal(dim(GRM[[1]]), c(4L, 4L))
    list(theta = c(1, 1), b = 2.5, Py = c(-1.5, -0.5, 0.5, 1.5), niter = 3)
  }
  first <- fit_fold(y, X, list(K), 5:6, fit_function = fake_fit)
  y[5:6] <- -1000
  second <- fit_fold(y, X, list(K), 5:6, fit_function = fake_fit)
  testthat::expect_equal(first$prediction, second$prediction)
  testthat::expect_false(identical(first$MSE, second$MSE))
})

testthat::test_that("native qgg predictions match the mixed-model formula", {
  set.seed(31)
  W <- matrix(rnorm(80 * 120), 80, 120)
  K <- tcrossprod(W) / ncol(W)
  X <- matrix(1, 80, 1)
  y <- as.numeric(W %*% rnorm(120) / sqrt(120) + rnorm(80, sd = 0.5))
  result <- fit_fold(y, X, list(K), 61:80)
  fit <- qgg::greml(y = y[1:60], X = X[1:60, , drop = FALSE],
                    GRM = list(K[1:60, 1:60]), maxit = 1000)
  V <- fit$theta[1] * K[1:60, 1:60] + diag(fit$theta[2], 60)
  expected <- as.numeric(X[61:80, , drop = FALSE] %*% fit$b +
                         fit$theta[1] * K[61:80, 1:60] %*% solve(V, y[1:60] - as.numeric(fit$b)))
  testthat::expect_equal(result$prediction, expected, tolerance = 1e-7)
})

testthat::test_that("evaluation writes every panel/family prediction with aligned IDs", {
  folder <- withr::local_tempdir()
  set.seed(31)
  ids <- c(paste0("F1E", 1:20), paste0("F2E", 1:20), paste0("A", 1:40))
  # Strong low-rank signal keeps variance components identifiable even in the
  # 20-sample LOFO training folds; an almost-identity G is unsuitable here.
  W <- matrix(rnorm(80 * 5), 80, 5)
  G <- tcrossprod(W) / ncol(W) + diag(0.05, 80)
  dimnames(G) <- list(ids, ids)
  Q <- matrix(rnorm(80 * 3), 80, 3, dimnames = list(ids, c("PC1", "PC2", "PC3")))
  y <- as.numeric(W %*% c(1, 2, -1, 0.5, 1) / sqrt(5) + rnorm(80, sd = 0.5))
  nam <- data.frame(PH = y[1:40], nam_only_metadata = 1,
                    row.names = paste0("NAM_H.", ids[1:40]))
  ames <- data.frame(PH = y[41:80], ames_only_metadata = "x",
                     row.names = paste0("Ames_H.", ids[41:80]))
  saveRDS(G, file.path(folder, "G.rds"))
  saveRDS(Q, file.path(folder, "Q.rds"))
  saveRDS(nam, file.path(folder, "nam.rds"))
  saveRDS(ames, file.path(folder, "ames.rds"))
  evaluate(file.path(folder, "G.rds"), file.path(folder, "Q.rds"),
            file.path(folder, "nam.rds"), file.path(folder, "ames.rds"),
            "NONE", "baseline", file.path(folder, "metrics.parquet"),
            file.path(folder, "predictions.parquet"), file.path(folder, "fit.json"), "PH")
  metrics <- as.data.frame(arrow::read_parquet(file.path(folder, "metrics.parquet")))
  predictions <- as.data.frame(arrow::read_parquet(file.path(folder, "predictions.parquet")))
  testthat::expect_equal(nrow(metrics), 4)
  testthat::expect_equal(nrow(predictions), 120)
  testthat::expect_setequal(predictions$sample_id[predictions$validation == "CP"], ids)
  testthat::expect_setequal(predictions$sample_id[predictions$validation == "LOFO"], ids[1:40])
  testthat::expect_true(all(is.finite(predictions$predicted)))
  report <- jsonlite::read_json(file.path(folder, "fit.json"))
  testthat::expect_equal(length(report$fits), 4)
})
