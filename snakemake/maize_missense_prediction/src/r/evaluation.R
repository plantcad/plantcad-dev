#' Normalize sample IDs and reject collisions.
#' @param ids Atomic vector coercible to unique, nonmissing character IDs.
#' @return Character vector with anchored panel prefixes removed.
normalize_ids <- function(ids) {
  ids <- sub("^(NAM_H|Ames_H)\\.", "", as.character(ids))
  stopifnot(!anyNA(ids), all(nzchar(ids)), !anyDuplicated(ids))
  ids
}

#' Align a named kernel to a specified sample order.
#' @param G Finite numeric square matrix with unique row and column names.
#' @param ids Unique character vector of required sample IDs.
#' @return Finite symmetric numeric matrix with rows/columns ordered as ids.
align_kernel <- function(G, ids) {
  stopifnot(is.matrix(G), !is.null(rownames(G)), !is.null(colnames(G)))
  rownames(G) <- normalize_ids(rownames(G))
  colnames(G) <- normalize_ids(colnames(G))
  stopifnot(all(ids %in% rownames(G)), all(ids %in% colnames(G)))
  G <- G[ids, ids, drop = FALSE]
  stopifnot(all(is.finite(G)), isTRUE(all.equal(G, t(G), tolerance = 1e-8)))
  G
}

#' Construct fixed cross-panel or leave-one-family-out folds.
#' @param ids Unique character vector of sample IDs.
#' @param panels Character vector of equal length: NAM_H or Ames_H.
#' @param validation Character scalar: CP or LOFO.
#' @return Named list of disjoint integer vectors of held-out sample indices.
validation_folds <- function(ids, panels, validation) {
  stopifnot(length(ids) == length(panels), !anyDuplicated(ids))
  if (validation == "CP") {
    stopifnot(setequal(unique(panels), c("NAM_H", "Ames_H")))
    return(split(seq_along(ids), panels))
  }
  stopifnot(validation == "LOFO", all(panels == "NAM_H"))
  # Family convention inherited from PR #29; save explicit membership with predictions.
  family <- sub("E.+$", "", ids)
  stopifnot(all(grepl("E", ids)), length(unique(family)) > 1)
  split(seq_along(ids), family)
}

#' Fit variance components on training samples and predict held-out samples.
#' @param y Finite numeric outcome vector.
#' @param X Finite numeric design matrix, with length(y) rows.
#' @param kernels List of finite symmetric numeric matrices, length(y) square.
#' @param heldout Unique integer vector of at least two validation indices.
#' @param maxit Positive integer scalar: REML iteration limit.
#' @param tolerance Positive numeric scalar: requested REML tolerance.
#' @param fit_function Function with qgg::greml's argument/result contract.
#' @return List: numeric prediction, observed, r, MSE, theta; integer n_train,
#'   n_test, niter; character warnings.
fit_fold <- function(y, X, kernels, heldout, maxit = 1000L, tolerance = 1e-5,
                     fit_function = qgg::greml) {
  stopifnot(length(heldout) >= 2, !anyDuplicated(heldout),
            all(heldout %in% seq_along(y)), all(is.finite(y)), all(is.finite(X)))
  stopifnot(is.numeric(y), is.matrix(X), is.numeric(X), nrow(X) == length(y),
            is.list(kernels), length(kernels) > 0,
            length(maxit) == 1L, maxit > 0, length(tolerance) == 1L, tolerance > 0)
  for (G in kernels) {
    stopifnot(is.matrix(G), is.numeric(G), all(dim(G) == length(y)),
              all(is.finite(G)), isTRUE(all.equal(G, t(G), tolerance = 1e-8)))
  }
  train <- setdiff(seq_along(y), heldout)
  stopifnot(length(train) > ncol(X), qr(X[train, , drop = FALSE])$rank == ncol(X))
  messages <- character()
  # Fit each fold explicitly: qgg's bundled CV wrapper does not forward maxit/tol
  # consistently. Validation phenotypes never enter variance-component fitting.
  fit <- withCallingHandlers(fit_function(
    y = y[train], X = X[train, , drop = FALSE],
    GRM = lapply(kernels, function(G) G[train, train, drop = FALSE]),
    maxit = maxit, tol = tolerance, ncores = 1, verbose = FALSE
  ), warning = function(w) {
    messages <<- c(messages, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
  stopifnot(length(fit$theta) == length(kernels) + 1,
            all(is.finite(fit$theta)), all(fit$theta >= 0),
            is.finite(fit$niter), fit$niter < maxit)
  prediction <- as.numeric(X[heldout, , drop = FALSE] %*% fit$b)
  for (j in seq_along(kernels)) {
    prediction <- prediction + as.numeric(
      kernels[[j]][heldout, train, drop = FALSE] %*% fit$Py * fit$theta[j]
    )
  }
  stopifnot(all(is.finite(prediction)))
  observed <- y[heldout]
  r <- if (sd(prediction) > 0 && sd(observed) > 0) cor(prediction, observed) else NA_real_
  list(prediction = prediction, observed = observed, r = r,
        MSE = mean((prediction - observed)^2), n_train = length(train),
        n_test = length(heldout), niter = fit$niter, theta = fit$theta, warnings = messages)
}

#' Evaluate every predefined trait and validation fold for one kernel setting.
#' @param g_path,q_path,nam_path,ames_path Character scalars: input RDS paths.
#' @param kernel_path Character scalar: additional kernel RDS path or NONE.
#' @param model Character scalar: result label.
#' @param metrics_path,predictions_path Character scalars: Parquet destinations.
#' @param report_path Character scalar: fit/provenance JSON destination.
#' @param traits Character vector of phenotype column names.
#' @param maxit Positive integer scalar: REML iteration limit.
#' @param tolerance Positive numeric scalar: requested REML tolerance.
#' @return Invisible named list of per-fold metric data frames.
evaluate <- function(g_path, q_path, nam_path, ames_path, kernel_path, model,
                     metrics_path, predictions_path, report_path, traits,
                     maxit = 1000L, tolerance = 1e-5) {
  G0 <- readRDS(g_path)
  Q <- readRDS(q_path)
  rownames(Q) <- normalize_ids(rownames(Q))
  X <- cbind("(Intercept)" = 1, as.matrix(Q[, c("PC1", "PC2", "PC3")]))
  panels <- list(NAM_H = as.data.frame(readRDS(nam_path))[, traits, drop = FALSE],
                  Ames_H = as.data.frame(readRDS(ames_path))[, traits, drop = FALSE])
  for (panel in names(panels)) {
    ids <- normalize_ids(rownames(panels[[panel]]))
    panels[[panel]]$sample_id <- ids
    panels[[panel]]$panel <- panel
    stopifnot(all(ids %in% rownames(X)))
  }
  CP <- do.call(rbind, panels)
  stopifnot(!anyDuplicated(CP$sample_id))
  raw_kernels <- list(G0 = G0)
  if (kernel_path != "NONE") raw_kernels$G1 <- readRDS(kernel_path)
  metrics <- predictions <- fits <- list()
  for (validation in c("CP", "LOFO")) {
    pheno <- if (validation == "CP") CP else panels$NAM_H
    for (trait in traits) {
      stopifnot(trait %in% names(pheno))
      data <- pheno[is.finite(pheno[[trait]]), , drop = FALSE]
      ids <- data$sample_id
      y <- setNames(data[[trait]], ids)
      design <- X[ids, , drop = FALSE]
      kernels <- lapply(raw_kernels, align_kernel, ids = ids)
      folds <- validation_folds(ids, data$panel, validation)
      for (fold in names(folds)) {
        message(model, " ", validation, " ", trait, " ", fold)
        result <- fit_fold(y, design, kernels, folds[[fold]], maxit, tolerance)
        key <- paste(validation, trait, fold, sep = "/")
        metrics[[key]] <- data.frame(model = model, validation = validation, trait = trait,
                                     fold = fold, n_train = result$n_train, n_test = result$n_test,
                                     r = result$r, MSE = result$MSE, niter = result$niter)
        predictions[[key]] <- data.frame(model = model, validation = validation, trait = trait,
                                         fold = fold, sample_id = ids[folds[[fold]]],
                                         observed = result$observed, predicted = result$prediction)
        fits[[key]] <- list(theta = result$theta, niter = result$niter, warnings = result$warnings)
      }
    }
  }
  arrow::write_parquet(do.call(rbind, metrics), metrics_path)
  arrow::write_parquet(do.call(rbind, predictions), predictions_path)
  jsonlite::write_json(list(model = model, fits = fits,
                            R_version = R.version.string,
                            qgg_version = as.character(utils::packageVersion("qgg")),
                            maxit = maxit, requested_tolerance = tolerance,
                            session = capture.output(sessionInfo())),
                       report_path, auto_unbox = TRUE, pretty = TRUE)
  invisible(metrics)
}
