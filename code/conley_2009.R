# =====================================================================
# Conley (spatial-HAC) standard errors for the 4 main 2009 PPML specs.
#
# Package: conleyreg (v0.1.9).  IMPORTANT limitation discovered in source:
#   conleyreg's panel handling (unit/time args) is OLS-ONLY. For a Poisson
#   model it stops with:
#     "Logit, probit, and poisson currently exclusively applicable to
#      cross-sectional data".
#   Fixed effects via the formula (| cell_id + yq) ARE supported for Poisson
#   (regressors are partialled by fixest::demean; point estimates come from
#   fixest::fepois, so they equal the paper's baseline exactly).
#
# Consequence / what this script actually does:
#   We stack the 81-cell x 48-quarter panel and run CROSS-SECTIONAL Conley
#   Poisson with cell + year-quarter fixed effects. Every observation carries
#   its cell centroid (lon,lat). The 48 quarterly observations of a given cell
#   share identical coordinates (distance 0 -> Bartlett weight 1), so within-
#   cell dependence across ALL quarters is fully absorbed into the spatial
#   sandwich (it behaves like a cell cluster), and spatial correlation across
#   DISTINCT cells within the cutoff is layered on top. No separate time-lag
#   (serial, lag_cutoff) decay is applied -- lag_cutoff is unavailable for the
#   Poisson model in conleyreg.
#
# Distances: spherical (Haversine), cutoffs 100 km and 200 km, Bartlett kernel.
#
# Reproduces baseline coefs (+0.136/+0.132/+0.155/+0.149) and the R fixest
# cell-clustered SEs (cross-language check vs the paper's pyfixest results),
# then reports Conley SE + p at 100 & 200 km for each spec.
#
# Inputs : data/generated/conley_panel_2009.csv  (built by code/conley_panel.py)
# Outputs: output/csv/conley_2009.csv
#          output/tables/tab_conley_2009.tex
#
# Standalone robustness diagnostic. NOT part of the numbered pipeline.
# Rscript: C:/Users/fcanda01/AppData/Local/Programs/R/R-4.5.1/bin/Rscript.exe
# =====================================================================

suppressMessages({
  library(conleyreg)
  library(data.table)
  library(fixest)
})

BASE     <- "C:/Users/fcanda01/Desktop/Recherche/kelp"
PANEL    <- file.path(BASE, "data", "generated", "conley_panel_2009.csv")
OUT_CSV  <- file.path(BASE, "output", "csv", "conley_2009.csv")
OUT_TEX  <- file.path(BASE, "output", "tables", "tab_conley_2009.tex")

dt <- fread(PANEL)
stopifnot(nrow(dt) == 3880L, uniqueN(dt$cell_id) == 81L, uniqueN(dt$yq) == 48L)
cat(sprintf("Panel: %d obs, %d cells, %d periods\n",
            nrow(dt), uniqueN(dt$cell_id), uniqueN(dt$yq)))

CUTOFFS  <- c(100, 200)
CONTROLS <- "ln_sst + ln_precipitation + ln_swh"

# 4 specs: {biomass,area} x {distance,count}
specs <- list(
  list(dep = "biomass_sum", exp = "ln_mean_length_2009_jan_jun_20km",
       out = "Biomass", trt = "Distance"),
  list(dep = "area_sum",    exp = "ln_mean_length_2009_jan_jun_20km",
       out = "Area",    trt = "Distance"),
  list(dep = "biomass_sum", exp = "ln_mean_count_2009_jan_jun_20km",
       out = "Biomass", trt = "Count"),
  list(dep = "area_sum",    exp = "ln_mean_count_2009_jan_jun_20km",
       out = "Area",    trt = "Count")
)

pstars <- function(p) {
  if (is.na(p)) return("")
  if (p < 0.01) return("***")
  if (p < 0.05) return("**")
  if (p < 0.10) return("*")
  ""
}

res <- vector("list", length(specs))

for (i in seq_along(specs)) {
  s <- specs[[i]]
  # Explicit interaction: policy_eca_2009 is 0/1, so the product equals the
  # exposure:policy interaction. A plain column (not a formula ":") keeps
  # conleyreg's internal fixest::demean of the regressors well-defined.
  dt[, treat_post := get(s$exp) * policy_eca_2009]

  f <- as.formula(
    sprintf("%s ~ treat_post + %s | cell_id + yq", s$dep, CONTROLS))

  # (1) Baseline: fixest fepois with cell-clustered SE  (R triangulation)
  m_cl  <- fepois(f, data = dt, vcov = ~cell_id)
  b     <- as.numeric(coef(m_cl)["treat_post"])
  se_cl <- as.numeric(se(m_cl)["treat_post"])
  p_cl  <- as.numeric(pvalue(m_cl)["treat_post"])
  Nobs  <- m_cl$nobs

  # (2) Conley cross-sectional Poisson at each cutoff
  conley <- lapply(CUTOFFS, function(cut) {
    ct <- conleyreg(f, data = dt, dist_cutoff = cut, model = "poisson",
                    lat = "lat", lon = "lon", dist_comp = "spherical",
                    kernel = "bartlett", ncores = 1, verbose = FALSE)
    list(coef = ct["treat_post", "Estimate"],
         se   = ct["treat_post", "Std. Error"],
         p    = ct["treat_post", ncol(ct)])   # last col = Pr(>|z|)/Pr(>|t|)
  })
  names(conley) <- paste0("c", CUTOFFS)

  # Sanity: conleyreg point estimate must equal the fepois baseline coef
  stopifnot(abs(conley$c100$coef - b) < 1e-8,
            abs(conley$c200$coef - b) < 1e-8)

  res[[i]] <- data.table(
    outcome     = s$out,
    exposure    = s$trt,
    coef        = b,
    N           = Nobs,
    cluster_se  = se_cl,       cluster_p  = p_cl,
    conley100_se = conley$c100$se, conley100_p = conley$c100$p,
    conley200_se = conley$c200$se, conley200_p = conley$c200$p
  )

  cat(sprintf(
    "%-8s x %-8s | b=%+.4f  clust=%.4f(p=%.4f)  C100=%.4f(p=%.4f)  C200=%.4f(p=%.4f)\n",
    s$out, s$trt, b, se_cl, p_cl,
    conley$c100$se, conley$c100$p, conley$c200$se, conley$c200$p))
}

R <- rbindlist(res)
fwrite(R, OUT_CSV)
cat("\nWrote", OUT_CSV, "\n")

# =====================================================================
# LaTeX table (booktabs + \sym), matching the paper's style.
# =====================================================================
fc <- function(x) formatC(x, format = "f", digits = 4)   # coef
fs <- function(x) sprintf("%.4f", x)                       # se

sym <- function(p) {
  st <- pstars(p)
  if (st == "") "" else sprintf("\\sym{%s}", st)
}

col <- function(row, field, pfield) {
  sprintf("%s%s", fc(row[[field]]), sym(row[[pfield]]))
}
se_cell <- function(open, close, row, field, pfield) {
  sprintf("%s%s%s%s", open, fs(row[[field]]), close, sym(row[[pfield]]))
}

L <- c()
L <- c(L, "\\def\\sym#1{\\ifmmode^{#1}\\else\\(^{#1}\\)\\fi}")
L <- c(L, "\\begin{tabular}{l*{4}{c}}")
L <- c(L, "\\toprule")
L <- c(L, "                    &\\multicolumn{2}{c}{Distance Traveled (km)}&\\multicolumn{2}{c}{Vessel Count}\\\\")
L <- c(L, "                    \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}")
L <- c(L, "                    &\\multicolumn{1}{c}{(1)}&\\multicolumn{1}{c}{(2)}&\\multicolumn{1}{c}{(3)}&\\multicolumn{1}{c}{(4)}\\\\")
L <- c(L, "                    &\\multicolumn{1}{c}{Biomass}&\\multicolumn{1}{c}{Area}&\\multicolumn{1}{c}{Biomass}&\\multicolumn{1}{c}{Area}\\\\")
L <- c(L, "\\midrule")

# Coefficient row (no stars on coef line; stars carried by each SE row)
coef_line <- "Post 2009 $\\times$ Exposure"
for (i in 1:4) coef_line <- sprintf("%s&%s", coef_line, fc(R$coef[i]))
L <- c(L, paste0(coef_line, "\\\\"))

# Cell-clustered SE row  ( ) with cluster stars
r1 <- "\\quad Cell-clustered SE"
for (i in 1:4) r1 <- sprintf("%s&%s", r1, se_cell("(", ")", R[i], "cluster_se", "cluster_p"))
L <- c(L, paste0(r1, "\\\\"))

# Conley 100 km SE row  [ ]
r2 <- "\\quad Conley SE (100 km)"
for (i in 1:4) r2 <- sprintf("%s&%s", r2, se_cell("[", "]", R[i], "conley100_se", "conley100_p"))
L <- c(L, paste0(r2, "\\\\"))

# Conley 200 km SE row  { }
r3 <- "\\quad Conley SE (200 km)"
for (i in 1:4) r3 <- sprintf("%s&%s", r3, se_cell("$\\{$", "$\\}$", R[i], "conley200_se", "conley200_p"))
L <- c(L, paste0(r3, "\\\\"))

L <- c(L, "\\midrule")
for (lab in c("Cell FE", "Year-Quarter FE", "Controls")) {
  ind <- lab
  for (i in 1:4) ind <- sprintf("%s&%s", ind, "Yes")
  L <- c(L, paste0(ind, "\\\\"))
}
obs <- "Observations"
for (i in 1:4) obs <- sprintf("%s&%d", obs, R$N[i])
L <- c(L, paste0(obs, "\\\\"))
L <- c(L, "\\bottomrule")
L <- c(L, "\\end{tabular}")

writeLines(L, OUT_TEX)
cat("Wrote", OUT_TEX, "\n")
