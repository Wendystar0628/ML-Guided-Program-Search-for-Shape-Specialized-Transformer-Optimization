# Complete deployment speedup beside all resident mechanism-family ablations.

.ablation_require_packages <- function() {
  required <- c("ggplot2", "patchwork", "scales")
  missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    stop(
      sprintf("Missing R package(s): %s", paste(missing, collapse = ", ")),
      call. = FALSE
    )
  }
}

.ablation_shape_labels <- c(
  "01" = "Shape 01\nB = 64",
  "02" = "Shape 02\nB = 1",
  "03" = "Shape 03\nB = 4",
  "04" = "Shape 04\nB = 16",
  "05" = "Shape 05\nB = 128",
  "06" = "Shape 06\nB = 10,000",
  "07" = "Shape 07\nD = 32",
  "08" = "Shape 08\nD = 1,024",
  "09" = "Shape 09\nH = 1",
  "10" = "Shape 10\nH = 2",
  "11" = "Shape 11\nH = 16",
  "12" = "Shape 12\nS = 32",
  "13" = "Shape 13\nS = 1,024"
)

.ablation_mechanism_labels <- c(
  runtime_schedule = "Runtime\nschedule",
  attention_path = "Attention\npath",
  layout_path = "Layout\npath",
  projection_precision = "Projection\nprecision",
  ffn_path = "FFN\npath",
  norm_boundary = "Norm /\nboundary"
)

.read_component_ablation <- function(data_path) {
  data <- utils::read.csv(
    data_path,
    stringsAsFactors = FALSE,
    check.names = FALSE,
    na.strings = c("", "NA")
  )
  report_validate_columns(
    data,
    c(
      "case_id",
      "mechanism_id",
      "status",
      "variant_kind",
      "ablation_slowdown",
      "retained_performance_fraction",
      "correctness_passed",
      "timed_samples_per_config"
    ),
    "component_ablation.csv"
  )

  data$shape <- report_shape_id(data$case_id)
  expected_shapes <- names(.ablation_shape_labels)
  expected_mechanisms <- names(.ablation_mechanism_labels)
  if (!setequal(unique(data$shape), expected_shapes)) {
    stop("Ablation table must contain every resident Shape 01-13.", call. = FALSE)
  }
  if (!setequal(unique(data$mechanism_id), expected_mechanisms)) {
    stop("Ablation table has an unexpected mechanism-family set.", call. = FALSE)
  }
  if (nrow(data) != length(expected_shapes) * length(expected_mechanisms)) {
    stop("Ablation table must contain one complete Shape-by-family grid.", call. = FALSE)
  }
  data$correctness_passed <- tolower(as.character(data$correctness_passed)) == "true"
  measured <- data$status == "measured"
  if (any(measured & !data$correctness_passed)) {
    stop("Measured ablation cells must pass correctness.", call. = FALSE)
  }
  if (any(measured & (!is.finite(data$ablation_slowdown) |
                      data$ablation_slowdown <= 0 |
                      !is.finite(data$retained_performance_fraction) |
                      data$retained_performance_fraction <= 0))) {
    stop("Measured slowdown and retained-performance values must be finite and positive.", call. = FALSE)
  }
  reciprocal_error <- abs(
    data$ablation_slowdown[measured] *
      data$retained_performance_fraction[measured] - 1
  )
  if (any(reciprocal_error > 1e-9)) {
    stop("Retained performance must be the reciprocal of ablation slowdown.", call. = FALSE)
  }
  data
}

.read_ablation_performance <- function(data_path) {
  data <- utils::read.csv(
    data_path,
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
      "speedup",
      "correctness_passed"
    ),
    "performance.csv"
  )
  data$shape <- report_shape_id(data$case_id)
  data <- data[data$shape %in% names(.ablation_shape_labels), , drop = FALSE]
  if (!setequal(data$shape, names(.ablation_shape_labels))) {
    stop("Performance table is missing one or more ablation Shapes.", call. = FALSE)
  }
  if (any(!is.finite(data$speedup) | data$speedup <= 1)) {
    stop("Complete-deployment speedups must be finite and above 1.", call. = FALSE)
  }
  data
}

draw_component_ablation <- function(
  ablation_path = report_source_data_path("component_ablation.csv"),
  performance_path = report_source_data_path("performance.csv"),
  base_size = 8.5,
  base_family = "Arial"
) {
  .ablation_require_packages()
  data <- .read_component_ablation(ablation_path)
  performance <- .read_ablation_performance(performance_path)
  shape_levels <- rev(unname(.ablation_shape_labels))

  performance$shape_label <- factor(
    unname(.ablation_shape_labels[performance$shape]),
    levels = shape_levels
  )
  performance$full_label <- sprintf("%.2fx", performance$speedup)

  total_panel <- ggplot2::ggplot(
    performance,
    ggplot2::aes(y = shape_label)
  ) +
    ggplot2::geom_vline(
      xintercept = 14.4926,
      colour = unname(REPORT_COLORS[["secondary"]]),
      linetype = "22",
      linewidth = 0.55
    ) +
    ggplot2::geom_segment(
      ggplot2::aes(x = 1, xend = speedup, yend = shape_label),
      colour = unname(REPORT_COLORS[["navy"]]),
      linewidth = 3.5,
      lineend = "butt"
    ) +
    ggplot2::geom_point(
      ggplot2::aes(x = speedup),
      shape = 21,
      size = 2.2,
      stroke = 0.6,
      fill = "white",
      colour = unname(REPORT_COLORS[["navy"]])
    ) +
    ggplot2::geom_text(
      ggplot2::aes(x = speedup, label = full_label),
      family = base_family,
      fontface = "bold",
      hjust = -0.18,
      size = 2.65,
      colour = unname(REPORT_COLORS[["ink"]])
    ) +
    ggplot2::scale_x_log10(
      limits = c(1, 96),
      breaks = c(1, 2, 4, 8, 16, 32, 64),
      labels = function(x) paste0(x, "x"),
      expand = ggplot2::expansion(mult = c(0, 0))
    ) +
    ggplot2::labs(
      title = "a  Complete deployment",
      subtitle = "Official baseline / deployed median",
      x = NULL,
      y = NULL
    ) +
    theme_report(base_size = base_size, base_family = base_family) +
    ggplot2::theme(
      axis.line.y = ggplot2::element_blank(),
      axis.ticks.y = ggplot2::element_blank(),
      axis.text.y = ggplot2::element_text(
        face = "bold",
        lineheight = 0.95,
        margin = ggplot2::margin(r = 5)
      ),
      panel.grid.major.y = ggplot2::element_blank(),
      panel.grid.major.x = ggplot2::element_line(
        colour = unname(REPORT_COLORS[["grid"]]),
        linewidth = 0.35
      ),
      plot.title = ggplot2::element_text(size = base_size, margin = ggplot2::margin(b = 2)),
      plot.subtitle = ggplot2::element_text(size = base_size - 1.3, margin = ggplot2::margin(b = 5)),
      plot.margin = ggplot2::margin(4, 4, 4, 4)
    )

  data$shape_label <- factor(
    unname(.ablation_shape_labels[data$shape]),
    levels = shape_levels
  )
  data$mechanism_label <- factor(
    unname(.ablation_mechanism_labels[data$mechanism_id]),
    levels = unname(.ablation_mechanism_labels)
  )
  data$fill_value <- ifelse(
    data$status == "measured",
    pmax(0, pmin(1.1, data$retained_performance_fraction)),
    NA_real_
  )
  suffix <- ifelse(
    data$variant_kind == "dependency_closure",
    "*",
    ifelse(data$variant_kind == "partial", "+", "")
  )
  data$cell_label <- ifelse(
    data$status == "measured",
    sprintf("%.0f%%%s", 100 * data$retained_performance_fraction, suffix),
    ifelse(
      data$status == "capacity_excluded",
      "CAP",
      ifelse(data$status == "not_isolatable", "CPL", "N/A")
    )
  )
  data$label_colour <- ifelse(
    data$status == "measured" &
      (data$retained_performance_fraction < 0.23 |
       data$retained_performance_fraction > 1.07),
    "white",
    unname(REPORT_COLORS[["ink"]])
  )

  mechanism_panel <- ggplot2::ggplot(
    data,
    ggplot2::aes(x = mechanism_label, y = shape_label, fill = fill_value)
  ) +
    ggplot2::geom_tile(
      width = 0.92,
      height = 0.82,
      colour = "white",
      linewidth = 0.75
    ) +
    ggplot2::geom_text(
      ggplot2::aes(label = cell_label, colour = label_colour),
      family = base_family,
      fontface = "bold",
      size = 2.55,
      show.legend = FALSE
    ) +
    ggplot2::scale_colour_identity() +
    ggplot2::scale_fill_gradientn(
      colours = c(
        unname(REPORT_COLORS[["orange"]]),
        "#F1C39D",
        "#F8F9F9",
        unname(REPORT_COLORS[["teal"]])
      ),
      values = scales::rescale(c(0, 0.5, 1, 1.1)),
      limits = c(0, 1.1),
      breaks = c(0, 0.25, 0.5, 0.75, 1),
      labels = c("0%", "25%", "50%", "75%", "100%"),
      oob = scales::squish,
      na.value = unname(REPORT_COLORS[["pale"]]),
      name = "Performance retained after removal"
    ) +
    ggplot2::guides(
      fill = ggplot2::guide_colourbar(
        direction = "horizontal",
        title.position = "top",
        title.hjust = 0.5,
        barwidth = grid::unit(42, "mm"),
        barheight = grid::unit(3.2, "mm"),
        ticks = TRUE
      )
    ) +
    ggplot2::labs(
      title = "b  Performance retained after removal",
      subtitle = "Ablated speedup / complete speedup; 100% = no change",
      x = NULL,
      y = NULL
    ) +
    theme_report(base_size = base_size, base_family = base_family) +
    ggplot2::theme(
      axis.line = ggplot2::element_blank(),
      axis.ticks = ggplot2::element_blank(),
      axis.text.x = ggplot2::element_text(
        face = "bold",
        lineheight = 0.92,
        margin = ggplot2::margin(t = 4)
      ),
      axis.text.y = ggplot2::element_blank(),
      panel.grid = ggplot2::element_blank(),
      legend.position = "bottom",
      legend.justification = "center",
      plot.title = ggplot2::element_text(size = base_size, margin = ggplot2::margin(b = 2)),
      plot.subtitle = ggplot2::element_text(size = base_size - 1.3, margin = ggplot2::margin(b = 5)),
      plot.margin = ggplot2::margin(4, 4, 4, 4)
    )

  (total_panel + mechanism_panel +
    patchwork::plot_layout(widths = c(0.88, 1.78))) +
    patchwork::plot_annotation(
      title = "Full gains and performance retained after mechanism removal",
      subtitle = paste0(
        "Each cell is the fraction of complete speedup retained after replacing one mechanism family with its legal fallback"
      ),
      caption = paste0(
        "100% means no change; lower values indicate stronger dependence; above 100% means the fallback was faster.\n",
        "Dashed line: 13-Shape geometric mean (14.49x). * dependency closure; + QKV-side partial.\n",
        "CAP = unsafe full-batch counterfactual; CPL = active but coupled; N/A = already on fallback.\n",
        "25 calls/config (Shape 06: 15); effects are non-additive."
      ),
      theme = theme_report(base_size = base_size, base_family = base_family) +
        ggplot2::theme(
          plot.title = ggplot2::element_text(size = base_size + 2, face = "bold"),
          plot.subtitle = ggplot2::element_text(size = base_size - 0.5),
          plot.caption = ggplot2::element_text(
            size = base_size - 1.5,
            lineheight = 1.08,
            colour = unname(REPORT_COLORS[["secondary"]])
          ),
          plot.margin = ggplot2::margin(4, 5, 5, 5)
        )
    ) &
    ggplot2::theme(legend.position = "bottom")
}
