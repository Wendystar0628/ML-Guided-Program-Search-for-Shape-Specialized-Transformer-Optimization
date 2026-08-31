library(ggplot2)
library(ggrepel)
library(scales)

.shape14_script_dir <- local({
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) > 0L) {
    dirname(normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = FALSE))
  } else {
    source_file <- tryCatch(sys.frame(1L)$ofile, error = function(...) NULL)
    if (is.null(source_file)) normalizePath(getwd()) else dirname(normalizePath(source_file))
  }
})

if (!exists("theme_report", mode = "function") || !exists("REPORT_COLORS")) {
  source(file.path(.shape14_script_dir, "theme_report.R"))
}


draw_shape14_capacity <- function() {
  data_path <- report_source_data_path("shape14_memory.csv")
  memory <- read.csv(data_path, stringsAsFactors = FALSE, check.names = FALSE)

  required_columns <- c("label", "gib", "evidence_kind")
  missing_columns <- setdiff(required_columns, names(memory))
  if (length(missing_columns) > 0L) {
    stop(
      "shape14_memory.csv is missing required columns: ",
      paste(missing_columns, collapse = ", ")
    )
  }
  if (any(!is.finite(memory$gib)) || any(memory$gib <= 0)) {
    stop("All Shape 14 memory values must be finite and strictly positive.")
  }

  capacity_row <- memory[memory$evidence_kind == "capacity", , drop = FALSE]
  evidence <- memory[memory$evidence_kind != "capacity", , drop = FALSE]
  if (nrow(capacity_row) != 1L || nrow(evidence) != 3L) {
    stop("Expected one capacity row and three Shape 14 evidence rows.")
  }

  row_order <- c(
    "Measured streamed B2 peak",
    "Dense B2 score tensor",
    "Dense B32 score tensor"
  )
  row_labels <- c(
    "Streamed B2 peak\nmeasured allocation",
    "Dense B2 score tensor\nanalytical lower bound",
    "Dense B32 score tensor\nanalytical context"
  )
  if (!setequal(evidence$label, row_order)) {
    stop("Shape 14 evidence labels do not match the expected report schema.")
  }

  evidence$y <- match(evidence$label, rev(row_order))
  evidence$value_label <- sprintf("%s GiB", comma(evidence$gib, accuracy = 0.01))
  evidence$series <- ifelse(
    evidence$evidence_kind == "measured",
    "Measured streamed peak",
    "Analytical dense bound"
  )

  capacity <- capacity_row$gib[[1L]]
  x_min <- 3
  x_max <- max(1e5, max(evidence$gib) * 4)
  measured <- evidence[evidence$series == "Measured streamed peak", , drop = FALSE]
  analytical <- evidence[evidence$series == "Analytical dense bound", , drop = FALSE]
  evidence$label_x <- c(8.2, 1500, 16000)[match(evidence$label, row_order)]
  evidence$label_hjust <- c(0, 0, 1)[match(evidence$label, row_order)]

  ggplot(evidence, aes(x = gib, y = y)) +
    geom_segment(
      aes(x = x_min, xend = x_max, y = y, yend = y),
      colour = REPORT_COLORS[["grid"]],
      linewidth = 0.35,
      inherit.aes = FALSE
    ) +
    geom_segment(
      x = capacity,
      xend = capacity,
      y = 0.62,
      yend = 3.30,
      colour = REPORT_COLORS[["secondary"]],
      linewidth = 0.65,
      linetype = "22",
      inherit.aes = FALSE
    ) +
    geom_point(
      data = measured,
      colour = REPORT_COLORS[["teal"]],
      size = 3.4,
      shape = 16
    ) +
    geom_point(
      data = analytical,
      colour = REPORT_COLORS[["orange"]],
      fill = "white",
      size = 3.8,
      stroke = 1.05,
      shape = 23
    ) +
    geom_text(
      aes(x = label_x, label = value_label, colour = series, hjust = label_hjust),
      size = 3.0,
      show.legend = FALSE
    ) +
    annotate(
      "label",
      x = capacity,
      y = 3.58,
      label = sprintf("RTX 4080 capacity  %s GiB", comma(capacity, accuracy = 0.01)),
      hjust = -0.05,
      vjust = 0.5,
      family = "Arial",
      size = 2.75,
      colour = REPORT_COLORS[["secondary"]],
      fill = "white",
      linewidth = 0,
      label.padding = grid::unit(1.2, "mm")
    ) +
    scale_colour_manual(
      values = c(
        "Measured streamed peak" = REPORT_COLORS[["teal"]],
        "Analytical dense bound" = REPORT_COLORS[["orange"]]
      )
    ) +
    scale_x_log10(
      limits = c(x_min, x_max),
      breaks = c(10, 100, 1000, 10000, 100000),
      labels = label_number(big.mark = ",", accuracy = 1),
      expand = expansion(mult = c(0, 0))
    ) +
    scale_y_continuous(
      limits = c(0.52, 3.78),
      breaks = 3:1,
      labels = row_labels,
      expand = expansion(mult = c(0, 0))
    ) +
    coord_cartesian(clip = "on") +
    labs(
      title = "Streaming keeps Shape 14 within device capacity",
      subtitle = "Dense score-tensor storage exceeds the 15.99 GiB device limit by orders of magnitude.",
      x = "Allocated or analytical storage (GiB, log scale)",
      y = NULL
    ) +
    theme_report(base_size = 9, base_family = "Arial") +
    theme(
      panel.grid.major.x = element_line(
        colour = REPORT_COLORS[["grid"]],
        linewidth = 0.35
      ),
      panel.grid.minor.x = element_blank(),
      panel.grid.major.y = element_blank(),
      axis.line.y = element_blank(),
      axis.ticks.y = element_blank(),
      axis.text.y = element_text(
        colour = REPORT_COLORS[["ink"]],
        hjust = 1,
        lineheight = 0.95,
        margin = margin(r = 10)
      ),
      plot.title.position = "plot",
      plot.margin = margin(t = 13, r = 12, b = 8, l = 38)
    )
}
