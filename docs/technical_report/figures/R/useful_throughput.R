# Two-panel useful-throughput figure: per-Shape rate and speedup relationship.

.throughput_require_packages <- function() {
  required <- c("ggplot2", "ggrepel", "patchwork")
  missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    stop(
      sprintf("Missing R package(s): %s", paste(missing, collapse = ", ")),
      call. = FALSE
    )
  }
}

.read_useful_throughput_data <- function(performance_csv) {
  data <- utils::read.csv(
    performance_csv,
    stringsAsFactors = FALSE,
    check.names = FALSE,
    na.strings = c("", "NA")
  )
  report_validate_columns(
    data,
    c("case_id", "speedup", "estimated_achieved_tflops"),
    "performance.csv"
  )

  data$shape <- report_shape_id(data$case_id)
  data
}

draw_useful_throughput <- function(
  performance_csv = report_source_data_path("performance.csv"),
  base_size = 9,
  base_family = "Arial"
) {
  .throughput_require_packages()
  data <- .read_useful_throughput_data(performance_csv)
  data <- data[
    is.finite(data$estimated_achieved_tflops),
    ,
    drop = FALSE
  ]

  if (nrow(data) == 0 || any(data$estimated_achieved_tflops < 0)) {
    stop("Useful throughput must contain finite, non-negative values.", call. = FALSE)
  }

  data <- data[order(as.integer(data$shape)), , drop = FALSE]
  data$shape_axis <- factor(data$shape, levels = rev(data$shape))
  data$is_shape14 <- data$shape == "14"
  throughput_limit <- max(data$estimated_achieved_tflops) * 1.17

  throughput_plot <- ggplot2::ggplot(
    data,
    ggplot2::aes(y = shape_axis, x = estimated_achieved_tflops)
  ) +
    ggplot2::geom_segment(
      ggplot2::aes(x = 0, xend = estimated_achieved_tflops, yend = shape_axis),
      colour = unname(REPORT_COLORS[["grid"]]),
      linewidth = 0.7
    ) +
    ggplot2::geom_point(
      ggplot2::aes(colour = is_shape14, shape = is_shape14),
      size = 2.8,
      stroke = 0.65
    ) +
    ggplot2::geom_text(
      ggplot2::aes(label = sprintf("%.1f", estimated_achieved_tflops)),
      hjust = 0,
      nudge_x = throughput_limit * 0.014,
      family = base_family,
      size = 2.5,
      colour = unname(REPORT_COLORS[["ink"]])
    ) +
    ggplot2::scale_colour_manual(
      values = c(
        "FALSE" = unname(REPORT_COLORS[["navy"]]),
        "TRUE" = unname(REPORT_COLORS[["orange"]])
      ),
      guide = "none"
    ) +
    ggplot2::scale_shape_manual(
      values = c("FALSE" = 16, "TRUE" = 18),
      guide = "none"
    ) +
    ggplot2::scale_x_continuous(
      limits = c(0, throughput_limit),
      breaks = seq(0, 80, by = 20),
      expand = ggplot2::expansion(mult = c(0, 0))
    ) +
    ggplot2::labs(
      title = "Useful throughput by Shape",
      subtitle = paste(
        "Orange diamond: streamed Shape 14",
        "Official B32 I/O validation pending",
        sep = "\n"
      ),
      x = "Project-estimated useful throughput (TFLOP/s)",
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

  resident <- data[
    data$shape != "14" & is.finite(data$speedup),
    ,
    drop = FALSE
  ]
  if (nrow(resident) == 0) {
    stop("No resident Shape has both speedup and throughput values.", call. = FALSE)
  }

  relationship_plot <- ggplot2::ggplot(
    resident,
    ggplot2::aes(x = speedup, y = estimated_achieved_tflops)
  ) +
    ggplot2::geom_point(
      size = 2.7,
      fill = unname(REPORT_COLORS[["teal"]]),
      colour = "white",
      shape = 21,
      stroke = 0.55
    ) +
    ggrepel::geom_text_repel(
      ggplot2::aes(label = shape),
      seed = 20260831,
      family = base_family,
      size = 2.45,
      colour = unname(REPORT_COLORS[["ink"]]),
      box.padding = 0.35,
      point.padding = 0.3,
      min.segment.length = 0,
      segment.colour = unname(REPORT_COLORS[["secondary"]]),
      segment.size = 0.3,
      max.overlaps = Inf,
      max.time = 1.5
    ) +
    ggplot2::scale_x_continuous(
      breaks = seq(0, 40, by = 10),
      expand = ggplot2::expansion(mult = c(0.04, 0.08))
    ) +
    ggplot2::scale_y_continuous(
      breaks = seq(0, 80, by = 20),
      expand = ggplot2::expansion(mult = c(0.03, 0.08))
    ) +
    ggplot2::labs(
      title = "Speedup versus useful throughput",
      subtitle = "Resident Shapes 01-13; Shape 14 excluded",
      x = "Speedup (baseline median / deployed median)",
      y = "Project-estimated useful throughput (TFLOP/s)"
    ) +
    theme_report(base_size = base_size, base_family = base_family) +
    ggplot2::theme(
      panel.grid.major = ggplot2::element_line(
        colour = unname(REPORT_COLORS[["grid"]]),
        linewidth = 0.35
      )
    )

  patchwork::wrap_plots(
    throughput_plot,
    relationship_plot,
    nrow = 1,
    widths = c(0.9, 1.1)
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
