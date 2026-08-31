search_evidence_script_dir <- local({
  source_file <- tryCatch(sys.frame(1)$ofile, error = function(...) NULL)
  if (is.null(source_file)) {
    file.path(getwd(), "docs", "technical_report", "figures", "R")
  } else {
    dirname(normalizePath(source_file, winslash = "/", mustWork = FALSE))
  }
})

if (!exists("theme_report", mode = "function") || !exists("REPORT_COLORS")) {
  source(file.path(search_evidence_script_dir, "theme_report.R"))
}


draw_search_evidence <- function() {
  required_packages <- c("ggplot2", "patchwork", "scales")
  missing_packages <- required_packages[
    !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
  ]
  if (length(missing_packages) > 0L) {
    stop(
      "Missing required R packages: ",
      paste(missing_packages, collapse = ", "),
      call. = FALSE
    )
  }

  source_data_dir <- dirname(report_source_data_path("search_flow.csv"))
  flow <- utils::read.csv(
    file.path(source_data_dir, "search_flow.csv"),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  timing <- utils::read.csv(
    file.path(source_data_dir, "search_cycle_timing.csv"),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )

  flow_columns <- c(
    "screen_new_trials",
    "enhanced_entries",
    "formal_comparisons",
    "deployment_updates"
  )
  timing_columns <- c(
    "case_id",
    "planning_seconds",
    "screen_seconds",
    "enhanced_seconds",
    "formal_seconds"
  )
  missing_flow_columns <- setdiff(flow_columns, names(flow))
  missing_timing_columns <- setdiff(timing_columns, names(timing))
  if (length(missing_flow_columns) > 0L) {
    stop(
      "search_flow.csv is missing: ",
      paste(missing_flow_columns, collapse = ", "),
      call. = FALSE
    )
  }
  if (length(missing_timing_columns) > 0L) {
    stop(
      "search_cycle_timing.csv is missing: ",
      paste(missing_timing_columns, collapse = ", "),
      call. = FALSE
    )
  }

  stage_names <- c("Screen", "Enhanced", "Formal", "Deployment")
  stage_counts <- colSums(flow[flow_columns], na.rm = TRUE)
  if (any(!is.finite(stage_counts)) || any(stage_counts <= 0)) {
    stop("All search-stage counts must be finite and positive.", call. = FALSE)
  }
  if (any(diff(stage_counts) >= 0)) {
    stop("Search-stage counts must decrease at every promotion step.", call. = FALSE)
  }

  flow_plot_data <- data.frame(
    stage = factor(stage_names, levels = stage_names),
    stage_index = seq_along(stage_names),
    count = as.numeric(stage_counts),
    colour_key = c("navy", "teal", "orange", "ink")
  )
  retention_plot_data <- data.frame(
    stage_index = seq_len(length(stage_names) - 1L) + 0.5,
    count = sqrt(stage_counts[-length(stage_counts)] * stage_counts[-1L]),
    label = sprintf(
      "%.1f%% retained",
      100 * stage_counts[-1L] / stage_counts[-length(stage_counts)]
    )
  )

  p_flow <- ggplot2::ggplot(
    flow_plot_data,
    ggplot2::aes(x = stage_index, y = count)
  ) +
    ggplot2::geom_line(
      colour = REPORT_COLORS[["secondary"]],
      linewidth = 0.65
    ) +
    ggplot2::geom_point(
      ggplot2::aes(colour = colour_key),
      size = 3.5,
      stroke = 0
    ) +
    ggplot2::geom_text(
      ggplot2::aes(label = scales::comma(count)),
      family = "Arial",
      fontface = "bold",
      size = 3.0,
      vjust = -0.9,
      colour = REPORT_COLORS[["ink"]]
    ) +
    ggplot2::geom_label(
      data = retention_plot_data,
      ggplot2::aes(x = stage_index, y = count, label = label),
      inherit.aes = FALSE,
      family = "Arial",
      size = 2.45,
      linewidth = 0,
      label.padding = grid::unit(0.10, "lines"),
      fill = "white",
      colour = REPORT_COLORS[["secondary"]]
    ) +
    ggplot2::scale_colour_manual(
      values = REPORT_COLORS[c("navy", "teal", "orange", "ink")],
      guide = "none"
    ) +
    ggplot2::scale_x_continuous(
      breaks = seq_along(stage_names),
      labels = stage_names,
      expand = ggplot2::expansion(mult = c(0.08, 0.08))
    ) +
    ggplot2::scale_y_log10(
      breaks = c(1, 10, 100, 1000, 10000),
      labels = scales::label_comma(),
      expand = ggplot2::expansion(mult = c(0.07, 0.20))
    ) +
    ggplot2::labs(
      title = "Candidate retention across four resident cycles",
      x = NULL,
      y = "Stage entries (log scale)"
    ) +
    theme_report(base_size = 9, base_family = "Arial") +
    ggplot2::theme(
      panel.grid.major.y = ggplot2::element_line(
        colour = REPORT_COLORS[["grid"]],
        linewidth = 0.35
      ),
      panel.grid.major.x = ggplot2::element_blank(),
      panel.grid.minor = ggplot2::element_blank(),
      axis.ticks.x = ggplot2::element_blank()
    )

  duration_columns <- c(
    Planning = "planning_seconds",
    Screen = "screen_seconds",
    Enhanced = "enhanced_seconds",
    Formal = "formal_seconds"
  )
  duration_values <- as.matrix(timing[unname(duration_columns)])
  storage.mode(duration_values) <- "double"
  if (any(!is.finite(duration_values)) || any(duration_values < 0)) {
    stop("All stage durations must be finite and non-negative.", call. = FALSE)
  }
  total_seconds <- rowSums(duration_values)
  if (any(total_seconds <= 0)) {
    stop("Each Shape must have a positive total duration.", call. = FALSE)
  }

  shape_ids <- sub("^official_", "", timing$case_id)
  shape_levels <- rev(shape_ids)
  timing_plot_data <- do.call(
    rbind,
    lapply(seq_len(nrow(timing)), function(row_index) {
      data.frame(
        shape = factor(shape_ids[[row_index]], levels = shape_levels),
        stage = factor(names(duration_columns), levels = names(duration_columns)),
        share = as.numeric(duration_values[row_index, ]) / total_seconds[[row_index]],
        stringsAsFactors = FALSE
      )
    })
  )
  total_plot_data <- data.frame(
    shape = factor(shape_ids, levels = shape_levels),
    x = 1.018,
    label = scales::label_number(accuracy = 1, suffix = " s")(total_seconds)
  )

  p_timing <- ggplot2::ggplot(
    timing_plot_data,
    ggplot2::aes(x = share, y = shape, fill = stage)
  ) +
    ggplot2::geom_col(width = 0.70, colour = "white", linewidth = 0.25) +
    ggplot2::geom_text(
      data = total_plot_data,
      ggplot2::aes(x = x, y = shape, label = label),
      inherit.aes = FALSE,
      family = "Arial",
      size = 2.55,
      hjust = 0,
      colour = REPORT_COLORS[["ink"]]
    ) +
    ggplot2::scale_fill_manual(
      values = c(
        Planning = REPORT_COLORS[["secondary"]],
        Screen = REPORT_COLORS[["navy"]],
        Enhanced = REPORT_COLORS[["teal"]],
        Formal = REPORT_COLORS[["orange"]]
      ),
      breaks = names(duration_columns),
      drop = FALSE
    ) +
    ggplot2::scale_x_continuous(
      breaks = seq(0, 1, by = 0.25),
      labels = scales::label_percent(accuracy = 1),
      expand = c(0, 0)
    ) +
    ggplot2::coord_cartesian(xlim = c(0, 1.14), clip = "off") +
    ggplot2::labs(
      title = "Screen dominates the measured stage time",
      x = "Share of one complete cycle",
      y = "Official Shape",
      fill = NULL
    ) +
    theme_report(base_size = 9, base_family = "Arial") +
    ggplot2::theme(
      panel.grid.major.y = ggplot2::element_blank(),
      panel.grid.minor = ggplot2::element_blank(),
      legend.position = "top",
      legend.justification = "left",
      legend.box.margin = ggplot2::margin(0, 0, 1, 0),
      plot.margin = ggplot2::margin(4, 16, 4, 4)
    )

  p_flow / p_timing +
    patchwork::plot_layout(heights = c(0.90, 1.35)) +
    patchwork::plot_annotation(tag_levels = "a") &
    ggplot2::theme(
      plot.tag = ggplot2::element_text(
        family = "Arial",
        face = "bold",
        size = 10,
        colour = REPORT_COLORS[["ink"]]
      )
    )
}
