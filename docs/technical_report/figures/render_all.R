#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
if (length(file_arg) != 1) {
  stop("render_all.R must be executed with Rscript.", call. = FALSE)
}

figure_dir <- dirname(normalizePath(sub("^--file=", "", file_arg), winslash = "/"))
r_dir <- file.path(figure_dir, "R")

source(file.path(r_dir, "theme_report.R"), local = FALSE, chdir = TRUE)
REPORT_R_DIR <- r_dir

for (script in c(
  "architecture_overview.R",
  "performance_summary.R",
  "workload_figures.R",
  "useful_throughput.R",
  "deployed_program_matrix.R",
  "component_ablation.R",
  "shape14_capacity.R",
  "search_evidence.R"
)) {
  source(file.path(r_dir, script), local = FALSE, chdir = TRUE)
}

required_packages <- c("ggplot2", "patchwork", "ggrepel", "scales", "svglite", "ragg")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0) {
  stop(
    "Missing R figure packages. Run install_dependencies.R first: ",
    paste(missing_packages, collapse = ", "),
    call. = FALSE
  )
}

save_report_figure <- function(plot, stem, width_mm, height_mm, dpi = 600) {
  width_in <- width_mm / 25.4
  height_in <- height_mm / 25.4
  base <- file.path(figure_dir, stem)

  svglite::svglite(paste0(base, ".svg"), width = width_in, height = height_in)
  print(plot)
  grDevices::dev.off()

  grDevices::cairo_pdf(
    paste0(base, ".pdf"),
    width = width_in,
    height = height_in,
    family = "Arial"
  )
  print(plot)
  grDevices::dev.off()

  ragg::agg_png(
    paste0(base, ".png"),
    width = width_in,
    height = height_in,
    units = "in",
    res = dpi,
    background = "white"
  )
  print(plot)
  grDevices::dev.off()
}

save_report_figure(draw_performance_summary(), "performance_summary", 183, 112)
save_report_figure(draw_architecture_overview(), "architecture_overview", 183, 120)
save_report_figure(draw_workload_landscape(), "workload_landscape", 150, 112)
save_report_figure(draw_workload_sensitivity(), "workload_sensitivity", 183, 116)
save_report_figure(draw_useful_throughput(), "useful_throughput", 183, 112)
save_report_figure(
  draw_deployed_program_matrix(
    file.path(figure_dir, "source_data", "deployed_programs.csv")
  ),
  "deployed_program_matrix",
  183,
  116
)
save_report_figure(draw_component_ablation(), "component_ablation", 183, 170)
save_report_figure(draw_shape14_capacity(), "shape14_capacity", 183, 82)
save_report_figure(draw_search_evidence(), "search_evidence", 183, 132)

dot_candidates <- c(
  Sys.which("dot"),
  "C:/Program Files/Graphviz/bin/dot.exe"
)
dot_candidates <- dot_candidates[nzchar(dot_candidates) & file.exists(dot_candidates)]
if (length(dot_candidates) == 0) {
  stop("Graphviz 'dot' was not found. Install Graphviz before rendering diagrams.", call. = FALSE)
}
dot <- normalizePath(dot_candidates[[1]], winslash = "/")

render_dot <- function(stem) {
  input <- file.path(figure_dir, "diagrams", paste0(stem, ".dot"))
  outputs <- list(svg = "svg", pdf = "pdf", png = "png")
  for (format in names(outputs)) {
    extra <- if (format == "png") "-Gdpi=600" else character()
    status <- system2(
      dot,
      c(paste0("-T", outputs[[format]]), extra, input, "-o", file.path(figure_dir, paste0(stem, ".", format)))
    )
    if (!identical(status, 0L)) {
      stop("Graphviz failed while rendering ", stem, " as ", format, call. = FALSE)
    }
  }
}

render_dot("shape14_streaming")

message("Rendered ten technical-report figures in: ", figure_dir)
