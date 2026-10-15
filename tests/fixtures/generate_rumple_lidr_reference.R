# Reproduce the Jenness reference values used by the Python tests.
# From the repository root:
#   Rscript tests/fixtures/generate_rumple_lidr_reference.R
# Requires R and sp only. The committed fixture records their versions.
#
# This runs the same operations as lidR's rumple_index.matrix, without needing
# lidR or its point-cloud dependencies:
# https://github.com/r-lidar/lidR/blob/master/R/metrics_stdmetrics.R#L498-L502
#
# sp's C routine treats the FIRST matrix dimension as X, matching PyForestScan.
# Do not transpose these matrices. Raster packages commonly expose matrices in
# (Y, X) order; those matrices would need transposing before calling this code.
# https://github.com/edzer/sp/blob/master/src/surfaceArea.c
# https://github.com/edzer/sp/blob/master/R/surfaceArea.R
#
# Missing centers are excluded. Missing neighbors use the center's height;
# the raster border is repeated. This does not interpolate the missing centers.

suppressPackageStartupMessages(library(sp))

case <- function(name, chm, resolution) {
  cell_area <- prod(resolution)
  areas <- sp::surfaceArea(
    chm, cellx = resolution[1], celly = resolution[2], byCell = TRUE
  )
  scalar <- sp::surfaceArea(chm, resolution[1], resolution[2]) /
    (sum(!is.na(chm)) * cell_area)
  list(name = name, chm = chm, resolution = resolution,
       per_cell = areas / cell_area, scalar = scalar)
}

cases <- list(
  case("flat_rectangular", matrix(7, 3, 4), c(2, 3)),
  case("slope_rectangular", outer(0:3 * 2, 0:4 * 3,
       function(x, y) 10 + 0.3 * x + 0.4 * y), c(2, 3)),
  case("roughness_and_gap_rectangular", matrix(c(
       1, 2, 3, 5,
       4, NA, 6, 7,
       9, 8, 7, 2
       ), nrow = 3, byrow = TRUE), c(2, 3)),
  case("roughness_square", matrix(c(
       1, 2, 4,
       3, 10, 5,
       7, 6, 8
       ), nrow = 3, byrow = TRUE), c(1, 1)),
  case("all_missing", matrix(NA_real_, 2, 3), c(2, 3)),
  case("single_cell", matrix(7, 1, 1), c(2, 3)),
  case("single_x_cell", matrix(c(1, 4, NA, 7), 1, 4), c(2, 3)),
  case("single_y_cell", matrix(c(1, NA, 4, 9), 4, 1), c(2, 3)),
  case("isolated_center", matrix(c(
       NA, NA, NA,
       NA, 8, NA,
       NA, NA, NA
       ), nrow = 3, byrow = TRUE), c(2, 3))
)

# Base-R JSON output keeps the reference generator dependency-free apart from sp.
number <- function(x) ifelse(is.finite(x), sprintf("%.17g", x), "null")
vector_json <- function(x) paste0("[", paste(number(x), collapse = ", "), "]")
matrix_json <- function(x) paste0("[\n",
  paste(vapply(seq_len(nrow(x)), function(i)
    paste0("        ", vector_json(x[i, ])), character(1)), collapse = ",\n"),
  "\n      ]")
case_json <- function(x) paste0(
  "    {\n",
  '      "name": "', x$name, '",\n',
  '      "cell_resolution": ', vector_json(x$resolution), ",\n",
  '      "chm": ', matrix_json(x$chm), ",\n",
  '      "per_cell_rumple": ', matrix_json(x$per_cell), ",\n",
  '      "scalar_rumple": ', number(x$scalar), "\n",
  "    }"
)
output <- paste0(
  "{\n",
  '  "reference": "lidR raster rumple via sp::surfaceArea",\n',
  '  "r_version": "', R.version.string, '",\n',
  '  "sp_version": "', as.character(packageVersion("sp")), '",\n',
  '  "array_axes": ["X", "Y"],\n',
  '  "missing_value": "JSON null represents R NA or NaN",\n',
  '  "cases": [\n',
  paste(vapply(cases, case_json, character(1)), collapse = ",\n"),
  "\n  ]\n}"
)
args <- commandArgs(trailingOnly = TRUE)
path <- if (length(args)) args[1] else "tests/fixtures/rumple_lidr_reference.json"
writeLines(output, path)
cat("Wrote", path, "using", R.version.string,
    "and sp", as.character(packageVersion("sp")), "\n")
