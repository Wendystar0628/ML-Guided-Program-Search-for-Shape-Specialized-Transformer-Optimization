# Two-panel final-performance figure: resident speedup and latency reduction.

.performance_require_packages <- function() {
  required <- c("ggplot2", "patchwork")
  missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    stop(
      sprintf("Missing R package(s): %s", paste(missing, collapse = ", ")),
      call. = FALSE
    )
  }
}

.read_performance_summary_data <- function(performance_csv) {
  data <- utils::read.csv(
    performance_csv,
    stringsAsFactors = FALSE,
    check.names = FALSE,
    na.strings = c("", "NA")
  )
  report_validate_columns(
    data,
    c(
      "case_id",
      "baseline_median_ms",
      "deployed_median_ms",
      "deployed_p90_ms",
      "speedup"
    ),
    "performance.csv"
  )

  data$shape <- report_shape_id(data$case_id)
  data
}

draw_performance_summary <- function(
  performance_csv = report_source_data_path("performance.csv"),
  base_size = 9,
  base_family = "Arial"
) {
  .performance_require_packages()
  data <- .read_performance_summary_data(performance_csv)
  resident <- data[
    is.finite(data$speedup) &
      is.finite(data$baseline_median_ms) &
      is.finite(data$deployed_median_ms) &
      is.finite(data$deployed_p90_ms),
    ,
    drop = FALSE
  ]

  if (nrow(resident) == 0) {
    stop("performance.csv has no complete resident-Shape rows.", call. = FALSE)
  }
  if (any(resident$baseline_median_ms <= 0) ||
      any(resident$deployed_median_ms <= 0) ||
      any(resident$deployed_p90_ms <= 0)) {
    stop("Latency values must be strictly positive for the log scale.", call. = FALSE)
  }

  resident <- resident[order(as.integer(resident$shape)), , drop = FALSE]
  resident$shape_axis <- factor(
    resident$shape,
    levels = rev(resident$shape)
  )
  geomean_speedup <- exp(mean(log(resident$speedup)))
  speedup_limit <- max(resident$speedup) * 1.16

  speedup_plot <- ggplot2::ggplot(
    resident,
    ggplot2::aes(y = shape_axis, x = speedup)
  ) +
    ggplot2::geom_col(
      width = 0.64,
      fill = unname(REPORT_COLORS[["navy"]]),
      colour = NA
    ) +
    ggplot2::geom_vline(
      xintercept = geomean_speedup,
      colour = unname(REPORT_COLORS[["orange"]]),
      linewidth = 0.65,
      linetype = "22"
    ) +
    ggplot2::geom_text(
      ggplot2::aes(label = sprintf("%.1fx", speedup)),
      hjust = 0,
      nudge_x = speedup_limit * 0.012,
      size = 2.5,
      family = base_family,
      colour = unname(REPORT_COLORS[["ink"]])
    ) +
    ggplot2::scale_x_continuous(
      limits = c(0, speedup_limit),
      breaks = seq(0, 40, by = 10),
      expand = ggplot2::expansion(mult = c(0, 0))
    ) +
    ggplot2::labs(
      title = "Resident speedup",
      subtitle = sprintf(
        "Geomean %.2fx\nShape 14 has no dense baseline",
        geomean_speedup
      ),
      x = "Speedup\n(baseline median / deployed median)",
      y = "Shape"
    ) +
    theme_report(base_size = base_size, base_family = base_family) +
    ggplot2::theme(
      panel.grid.major.y = ggplot2::element_blank(),
      panel.grid.major.x = ggplot2::element_line(
        colour = unname(REPORT_COLORS[["grid"]]),
        linewidth = 0.35
      )
    )

  latency_plot <- ggplot2::ggplot(
    resident,
    ggplot2::aes(y = shape_axis)
  ) +
    ggplot2::geom_segment(
      ggplot2::aes(
        x = deployed_median_ms,
        xend = baseline_median_ms,
        yend = shape_axis
      ),
      colour = unname(REPORT_COLORS[["secondary"]]),
      linewidth = 0.55,
      alpha = 0.72
    ) +
    ggplot2::geom_point(
      ggplot2::aes(
        x = baseline_median_ms,
        colour = "Baseline median",
        shape = "Baseline median"
      ),
      size = 2.15
    ) +
    ggplot2::geom_point(
      ggplot2::aes(
        x = deployed_median_ms,
        colour = "Deployed median",
        shape = "Deployed median"
      ),
      size = 2.25
    ) +
    ggplot2::geom_point(
      ggplot2::aes(
        x = deployed_p90_ms,
        colour = "Deployed P90",
        shape = "Deployed P90"
      ),
      fill = NA,
      stroke = 0.75,
      size = 2.55
    ) +
    ggplot2::scale_colour_manual(
      name = "Latency statistic",
      values = c(
        "Baseline median" = unname(REPORT_COLORS[["ink"]]),
        "Deployed median" = unname(REPORT_COLORS[["teal"]]),
        "Deployed P90" = unname(REPORT_COLORS[["orange"]])
      ),
      breaks = c("Baseline median", "Deployed median", "Deployed P90")
    ) +
    ggplot2::scale_shape_manual(
      name = "Latency statistic",
      values = c(
        "Baseline median" = 16,
        "Deployed median" = 16,
        "Deployed P90" = 1
      ),
      breaks = c("Baseline median", "Deployed median", "Deployed P90")
    ) +
    ggplot2::scale_x_log10(
      breaks = c(0.1, 1, 10, 100, 1000),
      labels = c("0.1", "1", "10", "100", "1,000")
    ) +
    ggplot2::labs(
      title = "Latency reduction",
      subtitle = "Horizontal distance shows baseline to deployed median",
      x = "Latency (ms; log scale)",
      y = NULL
    ) +
    theme_report(base_size = base_size, base_family = base_family) +
    ggplot2::theme(
      panel.grid.major.y = ggplot2::element_blank(),
      panel.grid.major.x = ggplot2::element_line(
        colour = unname(REPORT_COLORS[["grid"]]),
        linewidth = 0.35
      ),
      legend.position = "top",
      legend.justification = "center",
      legend.title = ggplot2::element_blank(),
      legend.key.width = grid::unit(3.5, "mm")
    )

  patchwork::wrap_plots(
    speedup_plot,
    latency_plot,
    nrow = 1,
    widths = c(0.95, 1.05)
  ) +
    patchwork::plot_annotation(
      tag_levels = "a",
      theme = ggplot2::theme(
        plot.tag = ggplot2::element_text(
          family = base_family,
          face = "bold",
          size = base_size + 1,
          colour = unname(REPORT_COLORS[["ink"]])
        )
      )
    )
}
