#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(oligo)
  library(limma)
  library(Biobase)
  library(AnnotationDbi)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: Rscript scripts/geo_deg.R <cel_dir> <out_dir>")
}
cel_dir <- args[[1]]
out_dir <- args[[2]]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ---- discover CEL files
cel_files <- list.files(cel_dir, pattern = "\\.CEL(\\.gz)?$", full.names = TRUE, ignore.case = TRUE)
if (length(cel_files) == 0) stop("No CEL files found in: ", cel_dir)

message("Reading CELs (n=", length(cel_files), ") …")
raw <- read.celfiles(cel_files)  # will load pd.* automatically if installed
plat <- annotation(raw)          # e.g. "pd.hugene.2.1.st", "pd.hg.u133.plus.2"
message("Detected platform design: ", plat)

# ---- ensure platform package & choose transcript DB
# Map platform -> transcript DB package
pick_db <- function(plat) {
  if (grepl("hugene.*2\\.1", plat)) return("hugene21sttranscriptcluster.db")
  if (grepl("hugene.*1\\.0", plat)) return("hugene10sttranscriptcluster.db")
  if (grepl("hg_u133_plus_2|hg\\.u133\\.plus\\.2|u133.*plus.*2", plat)) return("hgu133plus2.db")
  stop("Don't know a transcript DB for platform: ", plat)
}
db_pkg <- pick_db(plat)
message("Using annotation DB: ", db_pkg)

suppressPackageStartupMessages(library(db_pkg, character.only = TRUE))

# ---- RMA
message("Background correcting / normalizing (RMA) …")
eset <- rma(raw, target = "core")  # core probesets for Gene ST; works fine for U133 too
expr <- Biobase::exprs(eset)

# ---- sample groups from filenames (expects ..._TI_<GROUP>_*.CEL.gz)
# Examples: GSM3904901_B1_TI_MCD_205.CEL.gz  -> group = MCD
bn <- basename(sampleNames(eset))
parts <- strsplit(bn, "_")
groups <- vapply(parts, function(x) {
  # find token after "TI" if present, else last token before numeric id
  idx <- which(toupper(x) == "TI")
  if (length(idx) == 1 && length(x) >= idx + 1) return(x[idx + 1])
  # fallback: last alpha chunk
  y <- gsub("\\.CEL(\\.gz)?$", "", x[length(x)])
  y <- gsub("[0-9]+$", "", y)
  if (nzchar(y)) return(y)
  return("UNK")
}, character(1))
groups <- factor(groups)
message("Groups (counts):")
print(table(groups))

if (nlevels(groups) < 2) stop("Need ≥2 groups (got: ", nlevels(groups), "). Check filename parsing.")

# ---- probeID -> gene symbol mapping
keys <- rownames(expr)
keytype <- if (grepl("hugene", db_pkg)) "PROBEID" else "PROBEID"   # both use PROBEID
cols <- c("SYMBOL")
map <- AnnotationDbi::select(get(db_pkg), keys = keys, keytype = keytype, columns = cols)
map <- map[!is.na(map$SYMBOL) & nzchar(map$SYMBOL), c("PROBEID","SYMBOL")]
if (nrow(map) == 0) stop("No probe->gene SYMBOL mappings found. Platform/db mismatch? (db=", db_pkg, ")")

# keep only mapped rows
expr <- expr[rownames(expr) %in% map$PROBEID, , drop = FALSE]
map <- map[match(rownames(expr), map$PROBEID), ]
stopifnot(identical(map$PROBEID, rownames(expr)))

# ---- collapse to gene (median of probesets per SYMBOL)
message("Collapsing to gene symbols …")
split_idx <- split(seq_len(nrow(expr)), map$SYMBOL)
gexpr <- vapply(split_idx, function(idx) {
  apply(expr[idx, , drop = FALSE], 2, median, na.rm = TRUE)
}, numeric(ncol(expr)))
gexpr <- t(gexpr)
rownames(gexpr) <- names(split_idx)

if (nrow(gexpr) == 0) stop("Collapsed matrix has 0 genes. Check mapping/DB.")

# ---- design & contrasts (each group vs mean of others)
design <- model.matrix(~ 0 + groups)
colnames(design) <- levels(groups)
fit0 <- lmFit(gexpr, design)

levs <- levels(groups)
for (g in levs) {
  others <- setdiff(levs, g)
  if (length(others) == 0) next
  contrast <- sprintf("%s - (%s)/%d", g, paste(others, collapse = " + "), length(others))
  message("Contrast: ", contrast)
  cm <- makeContrasts(contrasts = contrast, levels = design)
  fit <- eBayes(contrasts.fit(fit0, cm))
  tt <- topTable(fit, number = Inf, sort.by = "t")
  out <- data.frame(gene = rownames(tt), logFC = tt$logFC, padj = tt$adj.P.Val, check.names = FALSE)
  of <- file.path(out_dir, paste0("DE_", g, "_vs_rest.csv"))
  write.csv(out, of, row.names = FALSE)
  message("Wrote: ", of)
}

message("Done.")