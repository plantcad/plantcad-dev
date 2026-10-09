# Sourceable functions: no work or package attachment is performed at import.
#' Export GDS variant metadata in bounded batches.
#' @param gds_path Character scalar: existing SNPRelate GDS path.
#' @param output Character scalar: gzipped TSV destination.
#' @param chunk_size Positive integer scalar: rows per batch.
#' @return Invisible integer scalar: number of exported variants.
extract_gds_metadata <- function(gds_path, output, chunk_size = 100000L) {
  stopifnot(length(chunk_size) == 1L, is.finite(chunk_size), chunk_size >= 1,
            chunk_size == as.integer(chunk_size))
  gds <- SNPRelate::snpgdsOpen(gds_path)
  on.exit(SNPRelate::snpgdsClose(gds), add = TRUE)
  nodes <- lapply(c("snp.id", "snp.chromosome", "snp.position", "snp.allele"),
                  function(name) gdsfmt::index.gdsn(gds, name))
  n <- gdsfmt::objdesp.gdsn(nodes[[1]])$dim[1]
  dest <- gzfile(output, "wt")
  on.exit(close(dest), add = TRUE)
  for (start in seq.int(1L, n, by = chunk_size)) {
    count <- min(chunk_size, n - start + 1L)
    columns <- lapply(nodes, function(node) gdsfmt::read.gdsn(node, start = start, count = count))
    frame <- data.frame(snp_id = as.character(columns[[1]]),
                        chrom = as.character(columns[[2]]),
                        pos = columns[[3]], alleles = columns[[4]])
    write.table(frame, dest, sep = "\t", row.names = FALSE,
                col.names = start == 1L, quote = FALSE)
  }
  invisible(n)
}

#' Compute one weighted, centered genotype cross-product.
#' @param X Numeric matrix: samples by SNPs; dosages 0/1/2, NA or 3 (missing).
#' @param weights Nonnegative finite numeric vector, one entry per SNP.
#' @param scale_genotypes Logical scalar: divide centered columns by their SD.
#' @return List with numeric matrix kernel, numeric used_weight, and integer
#'   missing and monomorphic counts.
weighted_block <- function(X, weights, scale_genotypes = FALSE) {
  stopifnot(is.matrix(X), is.numeric(X), nrow(X) > 0, is.numeric(weights),
            is.logical(scale_genotypes), length(scale_genotypes) == 1L,
            !is.na(scale_genotypes))
  stopifnot(ncol(X) == length(weights), all(is.finite(weights)), all(weights >= 0))
  X[X == 3] <- NA
  missing <- sum(is.na(X))
  stopifnot(all(X[!is.na(X)] %in% c(0, 1, 2)))
  means <- colMeans(X, na.rm = TRUE)
  means[!is.finite(means)] <- 0
  X <- sweep(X, 2, means, "-")
  X[is.na(X)] <- 0  # Mean imputation after centering.
  variance <- colSums(X^2) / nrow(X)
  keep <- variance > 0 & weights > 0
  X <- X[, keep, drop = FALSE]
  w <- weights[keep]
  if (scale_genotypes) X <- sweep(X, 2, sqrt(variance[keep]), "/")
  X <- sweep(X, 2, sqrt(w), "*")
  list(kernel = tcrossprod(X), used_weight = sum(w),
        missing = missing, monomorphic = sum(variance == 0))
}

#' Build a sample kernel from explicitly identified GDS SNPs.
#' @param gds_path Character scalar: SNPRelate GDS file.
#' @param weights_path Character scalar: Parquet file with snp_id and weight.
#' @param output Character scalar: RDS destination for the named kernel matrix.
#' @param report Character scalar: JSON QC destination.
#' @param chunk_size Positive integer scalar: SNPs per genotype block.
#' @param scale_genotypes Logical scalar: standardize genotype variance.
#' @return Invisible numeric matrix: symmetric kernel with mean diagonal one.
build_kernel <- function(gds_path, weights_path, output, report,
                          chunk_size = 2000L, scale_genotypes = FALSE) {
  weights <- as.data.frame(arrow::read_parquet(weights_path))
  stopifnot(!anyDuplicated(weights$snp_id), all(is.finite(weights$weight)),
            all(weights$weight >= 0), sum(weights$weight) > 0, chunk_size > 0)
  weights <- weights[weights$weight > 0, ]
  gds <- SNPRelate::snpgdsOpen(gds_path)
  on.exit(SNPRelate::snpgdsClose(gds), add = TRUE)
  sample_ids <- as.character(gdsfmt::read.gdsn(gdsfmt::index.gdsn(gds, "sample.id")))
  snp_ids <- gdsfmt::read.gdsn(gdsfmt::index.gdsn(gds, "snp.id"))
  indices <- match(weights$snp_id, as.character(snp_ids))
  stopifnot(!anyNA(indices), !anyDuplicated(sample_ids))
  selected_ids <- snp_ids[indices]
  G <- matrix(0, length(sample_ids), length(sample_ids), dimnames = list(sample_ids, sample_ids))
  missing_count <- monomorphic_count <- used_weight <- 0
  for (start in seq.int(1L, nrow(weights), by = chunk_size)) {
    idx <- start:min(start + chunk_size - 1L, nrow(weights))
    block <- SNPRelate::snpgdsGetGeno(gds, snp.id = selected_ids[idx], snpfirstdim = FALSE,
                                     with.id = TRUE, verbose = FALSE)
    X <- block$genotype[match(sample_ids, as.character(block$sample.id)), , drop = FALSE]
    w <- weights$weight[match(as.character(block$snp.id), weights$snp_id)]
    contribution <- weighted_block(X, w, scale_genotypes)
    G <- G + contribution$kernel
    missing_count <- missing_count + contribution$missing
    monomorphic_count <- monomorphic_count + contribution$monomorphic
    used_weight <- used_weight + contribution$used_weight
  }
  stopifnot(used_weight > 0, all(is.finite(G)))
  trace_scale <- mean(diag(G))
  stopifnot(trace_scale > 0)
  G <- G / trace_scale
  # A weighted cross-product is PSD by construction; do not alter it with nearPD.
  saveRDS(G, output)
  jsonlite::write_json(list(n_samples = length(sample_ids), n_nonzero_variants = nrow(weights),
                            missing_genotypes = missing_count, monomorphic_variants = monomorphic_count,
                            used_weight = used_weight, trace_scale = trace_scale,
                            centered = TRUE, variance_scaled = scale_genotypes),
                       report, auto_unbox = TRUE, pretty = TRUE)
  invisible(G)
}
