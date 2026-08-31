cran_packages <- c(
  "ggplot2",
  "patchwork",
  "ggrepel",
  "scales",
  "svglite",
  "ragg",
  "dplyr",
  "tidyr"
)

user_library <- Sys.getenv("R_LIBS_USER")
if (!nzchar(user_library)) {
  user_library <- file.path(Sys.getenv("LOCALAPPDATA"), "R", "win-library", paste0(R.version$major, ".", strsplit(R.version$minor, "\\.")[[1]][1]))
}
dir.create(user_library, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(user_library, .libPaths()))

missing <- cran_packages[!vapply(cran_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0) {
  install.packages(missing, repos = "https://cloud.r-project.org", lib = user_library, dependencies = NA)
}

still_missing <- cran_packages[!vapply(cran_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(still_missing) > 0) {
  stop("Missing R packages after installation: ", paste(still_missing, collapse = ", "))
}

message("R figure dependencies are ready.")
