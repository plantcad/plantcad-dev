# Title     : Evaluate observed G1 kernels from a variant effect score
# Objective : Given CHROM/POS/score, build G1 at q in {0.9,0.99,0.999} and compute CP/LOFO r

suppressPackageStartupMessages({
  library(SNPRelate)
  library(data.table)
  library(Matrix)
  library(qgg)
})

#--------------------------------------------------------
# Config (edit model_name and score_file)
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

traits <- c("GY_adjusted", "PH", "DTS")
Q_variables <- c("PC1", "PC2", "PC3")
n_cores <- as.integer(Sys.getenv("N_THREADS", unset = "32"))

# User inputs
model_name <- Sys.getenv("MODEL_NAME", unset = "your_glm")
score_file <- Sys.getenv("SCORE_FILE", unset = "path/to/score.rds")
score_col  <- Sys.getenv("SCORE_COL",  unset = "score")

quantiles  <- c(0.9, 0.99, 0.999)
chunk_size <- 20000

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
stopifnot(file.exists(G_file), file.exists(Q_file), file.exists(gds_file), file.exists(score_file))
G0 <- readRDS(G_file)
Q  <- readRDS(Q_file)
X_base <- Q[, c("(Intercept)", Q_variables), drop = FALSE]

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
setnames(map, c("chromosome","position"), c("CHROM","POS"))
map[, CHROM := as.character(CHROM)]
setkey(map, CHROM, POS)

sc <- readRDS(score_file)
sc <- as.data.table(sc)
stopifnot(all(c("CHROM","POS", score_col) %in% colnames(sc)))
sc[, CHROM := as.character(CHROM)]
setkey(sc, CHROM, POS)

annot <- sc[map, nomatch = 0]
stopifnot(nrow(annot) > 0)

qs <- setNames(as.numeric(stats::quantile(annot[[score_col]], probs = quantiles, na.rm = TRUE)), paste0("q", quantiles))
W_den <- sapply(qs, function(th) sum(annot[[score_col]] >= th, na.rm = TRUE))
saveRDS(W_den, weight_file)

CV <- list(CP = NULL, LOFO = NULL)

for (qn in names(qs)) {
  th <- qs[[qn]]
  w <- as.numeric(annot[[score_col]] >= th)
  G1 <- accumulate_kernel(geno, annot$snp.id, w, step = chunk_size)
  G1 <- Matrix::nearPD(G1 / W_den[[qn]])$mat %>% as.matrix()
  var_name <- paste0(model_name, ".", qn)

  for (validation in c("CP", "LOFO")) {
    Y  <- Y_list[[validation]]
    VS <- VS_list[[validation]]

    for (trait in colnames(Y)) {
      obs   <- rownames(Y)[is.finite(Y[, trait])]
      y.obs <- Y[obs, trait]
      X.obs <- X_base[obs, ]
      G.obs <- list(G0[obs, obs], G1[obs, obs])

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

cat("Saved observed CV to ", cv_out_file, "\n", sep = "")
