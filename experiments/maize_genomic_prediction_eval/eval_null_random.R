# Title     : Build and evaluate reusable null GRMs (q in {0.9,0.99,0.999})
# Objective : Create MAF-matched random G1 kernels, save them, and compute CP(cross-panel)/LOFO(leave-one-family-out) r

suppressPackageStartupMessages({
  library(SNPRelate)
  library(data.table)
  library(Matrix)
  library(qgg)
})

#--------------------------------------------------------
# Config
#--------------------------------------------------------
wd <- "/workdir/jz963/Ramstein_SNPConstraintPrediction_2021"
setwd(wd)

hyb_folder <- "hybrids"
gds_file   <- file.path(hyb_folder, "AGPv4_hybrids.gds")
G_file     <- file.path(hyb_folder, "G.rds")
Q_file     <- file.path(hyb_folder, "Q.rds")

VS_folders <- c(
  "NAM_H"  = "NAM_H",
  "Ames_H" = "Ames_H"
)
NAM_folder <- VS_folders[["NAM_H"]]

traits     <- c("GY_adjusted", "PH", "DTS")
Q_variables <- c("PC1", "PC2", "PC3")

quantiles  <- c(0.9, 0.99, 0.999)
B          <- as.integer(Sys.getenv("N_NULL", unset = "10"))
maf_bins   <- 10
chunk_size <- 20000
n_cores    <- as.integer(Sys.getenv("N_THREADS", unset = "32"))

model_name <- "random"
weight_file <- file.path(hyb_folder, paste0("weight_list-", model_name, ".rds"))
cv_out_file <- file.path(hyb_folder, paste0("CV_", model_name, ".rds"))

#--------------------------------------------------------
# Helpers
#--------------------------------------------------------
accumulate_kernel <- function(geno, snp_ids, w, step = 20000, center_X = FALSE, scale_X = FALSE) {
  sid <- SNPRelate::snpgdsSummary(geno)$sample.id
  n <- length(sid)
  G <- matrix(0, n, n, dimnames = list(sid, sid))
  sel <- which(w != 0 & is.finite(w))
  if (!length(sel)) return(G)
  snp_ids <- snp_ids[sel]
  w       <- w[sel]
  for (i in seq(1, length(snp_ids), by = step)) {
    idx <- i:min(i + step - 1, length(snp_ids))
    X_list <- SNPRelate::snpgdsGetGeno(geno, snp.id = snp_ids[idx], snpfirstdim = FALSE, with.id = FALSE)
    X <- scale(X_list, center = center_X, scale = scale_X)
    G <- G + X %*% (w[idx] * t(X))
    rm(X_list, X); gc()
  }
  G
}

get_pred_obs <- function(v) {
  cn <- colnames(v)
  pred <- intersect(c("ypred", "yhat", "fitted"), cn)
  obs  <- intersect(c("yobs", "y"), cn)
  if (!length(pred) || !length(obs)) stop("Cannot find prediction/observation columns in validation table.")
  list(pred = pred[[1]], obs = obs[[1]])
}

#--------------------------------------------------------
# Inputs
#--------------------------------------------------------
stopifnot(file.exists(G_file), file.exists(Q_file), file.exists(gds_file))
G0 <- readRDS(G_file)
Q  <- readRDS(Q_file)
X_base <- cbind("(Intercept)"=1, as.matrix(Q[, Q_variables]))

Y_list <- list()
Y_list[["CP"]] <- do.call(rbind, lapply(VS_folders, function(f) {
  readRDS(file.path(f, "pheno.rds"))[, traits]
}))
Y_list[["LOFO"]] <- readRDS(file.path(NAM_folder, "pheno.rds"))[, traits]

VS_list <- list()
VS_list[["CP"]] <- lapply(VS_folders, function(f) rownames(readRDS(file.path(f, "pheno.rds"))))
VS_list[["LOFO"]] <- split(rownames(Y_list[["LOFO"]]), gsub("E.+", "", rownames(Y_list[["LOFO"]])))

geno <- snpgdsOpen(gds_file)
on.exit(snpgdsClose(geno))
map <- data.table(SNPRelate::snpgdsSNPList(geno))
setnames(map, c("chromosome","position"), c("chr","pos"))

fr <- SNPRelate::snpgdsSNPRateFreq(geno)
maf <- pmin(fr$MinorFreq, 1 - fr$MinorFreq)
map[, maf := maf]
N <- nrow(map)
qs <- setNames(quantiles, paste0("q", quantiles))
n_q <- setNames(as.integer(round((1 - quantiles) * N)), names(qs))

saveRDS(n_q, weight_file)

# Bin by MAF
map[, bin := cut(maf, breaks = quantile(maf, probs = seq(0, 1, length.out = maf_bins + 1), na.rm = TRUE),
                 include.lowest = TRUE, labels = FALSE)]
bin_ix <- split(seq_len(N), map$bin)
bin_sizes <- vapply(bin_ix, length, 1L)
bin_prop <- bin_sizes / sum(bin_sizes)

CV <- list(CP = NULL, LOFO = NULL)

for (k in seq_len(B)) {
  message("Null replicate k=", k)

  GRM_list <- list()
  for (qn in names(qs)) {
    target <- n_q[[qn]]
    take <- floor(bin_prop * target)
    remainder <- target - sum(take)
    if (remainder > 0) {
      extras <- sample(seq_along(take), remainder, replace = TRUE, prob = bin_prop)
      take[extras] <- take[extras] + 1L
    }
    sel <- unlist(mapply(function(ix, m) { if (m <= 0) integer(0) else sample(ix, m) },
                         bin_ix, as.list(take), SIMPLIFY = FALSE), use.names = FALSE)

    w <- rep(0, N); w[sel] <- 1
    G1 <- accumulate_kernel(geno, map$snp.id, w, step = chunk_size)
    var_name <- paste0(model_name, ".", qn)
    GRM_list[[var_name]] <- G1
  }

  # Save GRMs for this replicate
  out_grm <- file.path(hyb_folder, sprintf("GRM_list-%s-%d.rds", model_name, k))
  saveRDS(GRM_list, out_grm)

  # Evaluate each G1 across validations and traits
  for (var_name in names(GRM_list)) {
    G1_full <- Matrix::nearPD(GRM_list[[var_name]] / n_q[sub(paste0(model_name, "\."), "", var_name)])$mat %>% as.matrix()

    for (validation in c("CP", "LOFO")) {
      Y  <- Y_list[[validation]]
      VS <- VS_list[[validation]]

      for (trait in colnames(Y)) {
        obs   <- rownames(Y)[is.finite(Y[, trait])]
        y.obs <- Y[obs, trait]
        X.obs <- X_base[obs, ]
        G.obs <- list(G0[obs, obs], G1_full[obs, obs])

        VS.obs <- lapply(VS, function(cv_split) na.omit(match(cv_split, obs)))

        fit <- greml(y = y.obs,
                     X = X.obs,
                     GRM = G.obs,
                     validate = VS.obs,
                     ncores = n_cores,
                     maxit = 1000,
                     tol = 1e-10)

        r <- sapply(names(VS.obs), function(split) {
          v <- fit$validation[[split]]
          cols <- get_pred_obs(v)
          suppressWarnings(cor(v[, cols$pred], v[, cols$obs], use = "complete"))
        })

        out <- data.frame(
          model    = model_name,
          k        = k,
          var_name = var_name,
          trait    = trait,
          family   = names(VS.obs),
          r        = as.numeric(r),
          MSE      = fit$accuracy$MSPE,
          stringsAsFactors = FALSE
        )
        CV[[validation]] <- rbind(CV[[validation]], out)
        saveRDS(CV, cv_out_file)
      }
    }
  }
}

cat("Saved null CV to ", cv_out_file, "\n", sep = "")
