# Title     : Evaluate baseline model (G0 only)
# Objective : Compute CP and LOFO validation r using only polygenic GRM G0

suppressPackageStartupMessages({
  library(data.table)
  library(qgg)
  library(Matrix)
})

#--------------------------------------------------------
# Config
#--------------------------------------------------------
wd <- "/workdir/jz963/Ramstein_SNPConstraintPrediction_2021"
setwd(wd)

input_root <- "."                 # relative to wd
hyb_folder <- file.path(input_root, "hybrids")
VS_folders <- c(
  "NAM_H"  = file.path(input_root, "NAM_H"),
  "Ames_H" = file.path(input_root, "Ames_H")
)
NAM_folder <- VS_folders[["NAM_H"]]

traits <- c("GY_adjusted", "PH", "DTS")
Q_variables <- c("PC1", "PC2", "PC3")
n_cores <- as.integer(Sys.getenv("N_THREADS", unset = "32"))

G_file <- file.path(hyb_folder, "G.rds")
Q_file <- file.path(hyb_folder, "Q.rds")

out_file <- file.path(hyb_folder, "CV_baseline.rds")

#--------------------------------------------------------
# Helpers
#--------------------------------------------------------
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
stopifnot(file.exists(G_file), file.exists(Q_file))
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
    G.obs <- list(G0[obs, obs])

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

    out <- data.frame(
      model           = "baseline",
      trait           = trait,
      family          = names(VS.obs),
      r               = as.numeric(r),
      MSE             = fit$accuracy$MSPE,
      stringsAsFactors = FALSE
    )

    CV[[validation]] <- rbind(CV[[validation]], out)
    saveRDS(CV, out_file)
  }
}

cat("Saved baseline CV to ", out_file, "\n", sep = "")
