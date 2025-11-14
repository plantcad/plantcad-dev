# Title     : Evaluate score-based model
# Objective : Given variant scores, build weighted G1 and compute CP/LOFO r

suppressPackageStartupMessages({
  library(SNPRelate)
  library(data.table)
  library(Matrix)
  library(qgg)
  library(optparse)
  library(arrow)
})

#--------------------------------------------------------
# Parse command-line arguments
#--------------------------------------------------------
option_list <- list(
  make_option(c("--hybrids-dir"), type="character", default=NULL,
              help="Path to hybrids folder", metavar="character"),
  make_option(c("--nam-dir"), type="character", default=NULL,
              help="Path to NAM_H folder", metavar="character"),
  make_option(c("--ames-dir"), type="character", default=NULL,
              help="Path to Ames_H folder", metavar="character"),
  make_option(c("--gds-file"), type="character", default=NULL,
              help="Path to GDS file", metavar="character"),
  make_option(c("--scores-file"), type="character", default=NULL,
              help="Path to variant scores parquet file", metavar="character"),
  make_option(c("--output"), type="character", default=NULL,
              help="Output parquet file path", metavar="character"),
  make_option(c("--n-threads"), type="integer", default=as.integer(Sys.getenv("N_THREADS", unset = "32")),
              help="Number of threads [default: %default or N_THREADS env var]", metavar="integer")
)

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

# Validate required arguments
if (is.null(opt$`hybrids-dir`) || is.null(opt$`nam-dir`) || is.null(opt$`ames-dir`) ||
    is.null(opt$`gds-file`) || is.null(opt$`scores-file`) || is.null(opt$output)) {
  print_help(opt_parser)
  stop("Required arguments: --hybrids-dir, --nam-dir, --ames-dir, --gds-file, --scores-file, --output", call.=FALSE)
}

#--------------------------------------------------------
# Config
#--------------------------------------------------------
hyb_folder <- opt$`hybrids-dir`
VS_folders <- c(
  "NAM_H"  = opt$`nam-dir`,
  "Ames_H" = opt$`ames-dir`
)
NAM_folder <- VS_folders[["NAM_H"]]

traits <- c("GY_adjusted", "PH", "DTS")
Q_variables <- c("PC1", "PC2", "PC3")
n_cores <- opt$`n-threads`

G_file <- file.path(hyb_folder, "G.rds")
Q_file <- file.path(hyb_folder, "Q.rds")
gds_file <- opt$`gds-file`
scores_file <- opt$`scores-file`
out_file <- opt$output

chunk_size <- 20000

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
# Load inputs
#--------------------------------------------------------
stopifnot(file.exists(G_file), file.exists(Q_file), file.exists(gds_file), file.exists(scores_file))
G0 <- readRDS(G_file)
Q  <- readRDS(Q_file)
X_base <- cbind("(Intercept)"=1, as.matrix(Q[, Q_variables]))

# Phenotypes
Y_list <- list()
Y_list[["CP"]] <- do.call(rbind, lapply(VS_folders, function(f) {
  readRDS(file.path(f, "pheno.rds"))[, traits]
}))
Y_list[["LOFO"]] <- readRDS(file.path(NAM_folder, "pheno.rds"))[, traits]

# Validation sets
VS_list <- list()
VS_list[["CP"]] <- lapply(VS_folders, function(f) rownames(readRDS(file.path(f, "pheno.rds"))))
VS_list[["LOFO"]] <- split(rownames(Y_list[["LOFO"]]), gsub("E.+", "", rownames(Y_list[["LOFO"]])))

# Open GDS and get SNP list
geno <- snpgdsOpen(gds_file)
on.exit(snpgdsClose(geno))

map <- data.table(SNPRelate::snpgdsSNPList(geno))

# Read variant scores from parquet
scores_df <- read_parquet(scores_file)
stopifnot("score" %in% colnames(scores_df))
scores <- as.numeric(scores_df$score)

# Validate scores: assert all are non-negative
if (any(is.na(scores))) {
  stop("Scores contain NA values. All scores must be non-negative and finite.")
}
if (any(scores < 0)) {
  stop("Scores contain negative values. All scores must be non-negative.")
}
if (sum(scores) == 0) {
  stop("Sum of scores is zero. At least some scores must be positive for normalization.")
}

# Normalize scores to sum to 1
w <- scores / sum(scores)

# Ensure we have the same number of scores as SNPs
stopifnot(length(w) == nrow(map))

# Build weighted GRM using normalized scores
G1 <- accumulate_kernel(geno, map$snp.id, w, step = chunk_size)
G1 <- as.matrix(Matrix::nearPD(G1)$mat)

#--------------------------------------------------------
# Evaluate
#--------------------------------------------------------
CV <- list(CP = NULL, LOFO = NULL)

for (validation in c("CP", "LOFO")) {
  Y  <- Y_list[[validation]]
  rownames(Y) <- gsub(pattern = 'NAM_H.', replacement = '',
                      x = rownames(Y))
  rownames(Y) <- gsub(pattern = 'Ames_H.', replacement = '',
                      x = rownames(Y))

  VS <- VS_list[[validation]]

  for (trait in colnames(Y)) {
    obs   <- rownames(Y)[is.finite(Y[, trait])]
    y.obs <- Y[obs, trait]
    names(y.obs) <- obs
    X.obs <- X_base[obs, ]
    G.obs <- list(G0[obs, obs], G1[obs, obs])

    VS.obs <- lapply(VS,
                     function(cv_split) na.omit(match(cv_split, obs)))

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

    # Extract model name from scores file path (remove directory and .parquet extension)
    model_name <- sub("\\.parquet$", "", basename(scores_file))

    out <- data.frame(
      model           = model_name,
      trait           = trait,
      family          = names(VS.obs),
      r               = as.numeric(r),
      MSE             = fit$accuracy$MSPE,
      stringsAsFactors = FALSE
    )

    CV[[validation]] <- rbind(CV[[validation]], out)
  }
}

# Combine CP and LOFO into a single data frame with validation column
CV_combined <- rbind(
  cbind(validation = "CP", CV[["CP"]]),
  cbind(validation = "LOFO", CV[["LOFO"]])
)

# Save as Parquet
write_parquet(CV_combined, out_file)
