# Title     : Extract variant coordinates from GDS file
# Objective : Extract chrom, pos, ref, alt from GDS and output as Parquet

suppressPackageStartupMessages({
  library(SNPRelate)
  library(data.table)
  library(optparse)
  library(arrow)
})

#--------------------------------------------------------
# Parse command-line arguments
#--------------------------------------------------------
option_list <- list(
  make_option(c("--gds-file"), type="character", default=NULL,
              help="Path to GDS file", metavar="character"),
  make_option(c("--output"), type="character", default=NULL,
              help="Output Parquet file path", metavar="character")
)

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

# Validate required arguments
if (is.null(opt$`gds-file`) || is.null(opt$output)) {
  print_help(opt_parser)
  stop("Required arguments: --gds-file, --output", call.=FALSE)
}

gds_file <- opt$`gds-file`
out_file <- opt$output

#--------------------------------------------------------
# Extract variant information
#--------------------------------------------------------
stopifnot(file.exists(gds_file))

geno <- snpgdsOpen(gds_file)
on.exit(snpgdsClose(geno))

# Get SNP list from GDS file
snp_list <- SNPRelate::snpgdsSNPList(geno)
map <- data.table(snp_list)

# Parse allele field to extract ref and alt
# Allele format is typically "A/G" or "A,C" for multiallelic
# For biallelic, format is "A/G"; for multiallelic, format is "A,C,G"
# We'll take the first allele as ref and second as alt
allele_split <- strsplit(map$allele, split="/|,", perl=TRUE)
map[, ref := sapply(allele_split, function(x) x[1])]
map[, alt := sapply(allele_split, function(x) if(length(x) > 1) x[2] else NA_character_)]

# Create output data frame with required columns
variants <- data.frame(
  chrom = as.character(map$chromosome),
  pos = as.integer(map$position),
  ref = map$ref,
  alt = map$alt,
  stringsAsFactors = FALSE
)

# Remove rows with missing alt (shouldn't happen, but just in case)
variants <- variants[!is.na(variants$alt), ]

# Save as Parquet
write_parquet(variants, out_file)
