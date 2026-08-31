# Shared visual contract for technical-report R figures.

REPORT_COLORS <- c(
  navy = "#356C95",
  teal = "#1F8A78",
  orange = "#D9772B",
  ink = "#24313A",
  secondary = "#69747B",
  grid = "#DDE2E5",
  pale = "#EEF3F6"
)

.report_source_file <- tryCatch(
  normalizePath(sys.frame(1)$ofile, winslash = "/", mustWork = FALSE),
  error = function(...) ""
)

REPORT_R_DIR <- if (nzchar(.report_source_file)) {
  dirname(.report_source_file)
} else {
  ""
}

theme_report <- function(base_size = 9, base_family = "Arial") {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required by theme_report().", call. = FALSE)
  }

  ggplot2::theme_classic(base_size = base_size, base_family = base_family) +
    ggplot2::theme(
      text = ggplot2::element_text(
        family = base_family,
        colour = unname(REPORT_COLORS[["ink"]])
      ),
      plot.title = ggplot2::element_text(
        size = base_size + 1,
        face = "bold",
        colour = unname(REPORT_COLORS[["ink"]]),
        margin = ggplot2::margin(b = 3)
      ),
      plot.subtitle = ggplot2::element_text(
        size = base_size - 1.2,
        colour = unname(REPORT_COLORS[["secondary"]]),
        margin = ggplot2::margin(b = 6)
      ),
      plot.caption = ggplot2::element_text(
        size = base_size - 1.5,
        colour = unname(REPORT_COLORS[["secondary"]]),
        hjust = 0,
        margin = ggplot2::margin(t = 5)
      ),
      axis.title = ggplot2::element_text(
        size = base_size - 0.4,
        colour = unname(REPORT_COLORS[["ink"]])
      ),
      axis.text = ggplot2::element_text(
        size = base_size - 1.2,
        colour = unname(REPORT_COLORS[["ink"]])
      ),
      axis.line = ggplot2::element_line(
        colour = unname(REPORT_COLORS[["ink"]]),
        linewidth = 0.35,
        lineend = "square"
      ),
      axis.ticks = ggplot2::element_line(
        colour = unname(REPORT_COLORS[["ink"]]),
        linewidth = 0.3
      ),
      axis.ticks.length = grid::unit(1.5, "mm"),
      panel.grid.major = ggplot2::element_line(
        colour = unname(REPORT_COLORS[["grid"]]),
        linewidth = 0.35
      ),
      panel.grid.minor = ggplot2::element_blank(),
      legend.position = "top",
      legend.justification = "left",
      legend.title = ggplot2::element_blank(),
      legend.text = ggplot2::element_text(size = base_size - 1.2),
      legend.key.height = grid::unit(3.5, "mm"),
      legend.key.width = grid::unit(4.5, "mm"),
      legend.spacing.x = grid::unit(1.5, "mm"),
      strip.background = ggplot2::element_blank(),
      strip.text = ggplot2::element_text(
        size = base_size - 0.2,
        face = "bold",
        colour = unname(REPORT_COLORS[["ink"]])
      ),
      panel.spacing = grid::unit(5, "mm"),
      plot.margin = ggplot2::margin(5, 7, 5, 5)
    )
}

theme_report_void <- function(base_size = 9, base_family = "Arial") {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required by theme_report_void().", call. = FALSE)
  }

  ggplot2::theme_void(base_size = base_size, base_family = base_family) +
    ggplot2::theme(
      text = ggplot2::element_text(
        family = base_family,
        colour = unname(REPORT_COLORS[["ink"]])
      ),
      plot.title = ggplot2::element_text(
        size = base_size + 1,
        face = "bold",
        colour = unname(REPORT_COLORS[["ink"]]),
        margin = ggplot2::margin(b = 3)
      ),
      plot.subtitle = ggplot2::element_text(
        size = base_size - 1.2,
        colour = unname(REPORT_COLORS[["secondary"]]),
        margin = ggplot2::margin(b = 6)
      ),
      plot.margin = ggplot2::margin(5, 7, 5, 5)
    )
}

report_source_data_path <- function(filename) {
  candidates <- c(
    if (nzchar(REPORT_R_DIR)) file.path(dirname(REPORT_R_DIR), "source_data", filename),
    file.path(getwd(), "docs", "technical_report", "figures", "source_data", filename),
    file.path(getwd(), "source_data", filename),
    file.path(getwd(), filename)
  )
  candidates <- unique(candidates[nzchar(candidates)])
  hit <- candidates[file.exists(candidates)]

  if (length(hit) == 0) {
    stop(
      sprintf(
        "Could not locate source-data file '%s'. Checked: %s",
        filename,
        paste(candidates, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  normalizePath(hit[[1]], winslash = "/", mustWork = TRUE)
}

report_shape_id <- function(case_id) {
  sub("^.*_", "", as.character(case_id))
}

report_validate_columns <- function(data, required, context = "data") {
  missing <- setdiff(required, names(data))
  if (length(missing) > 0) {
    stop(
      sprintf(
        "%s is missing required column(s): %s",
        context,
        paste(missing, collapse = ", ")
      ),
      call. = FALSE
    )
  }
  invisible(data)
}
