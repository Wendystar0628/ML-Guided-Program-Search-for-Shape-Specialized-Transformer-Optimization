## Layered system overview for the shape-specialized optimization loop.
## One reading direction is used within each stage; the short return arrow is
## evidence feedback, while the downward path is deployment.

suppressPackageStartupMessages(library(ggplot2))

.architecture_script_dir <- local({
  source_file <- tryCatch(sys.frame(1)$ofile, error = function(...) NULL)
  if (is.null(source_file) || !nzchar(source_file)) {
    file.path(getwd(), "R")
  } else {
    dirname(normalizePath(source_file, winslash = "/", mustWork = FALSE))
  }
})

if (!exists("theme_report_void", mode = "function") || !exists("REPORT_COLORS")) {
  source(file.path(.architecture_script_dir, "theme_report.R"), local = FALSE)
}

.architecture_pt <- 72.27 / 25.4

.architecture_arrow <- function(colour = "#89939A", length_mm = 1.55) {
  grid::arrow(
    angle = 22,
    length = grid::unit(length_mm, "mm"),
    ends = "last",
    type = "closed"
  )
}

draw_architecture_overview <- function() {
  bands <- data.frame(
    xmin = c(2, 2, 2),
    xmax = c(98, 98, 98),
    ymin = c(66, 36, 3),
    ymax = c(97, 62, 32),
    fill = c("#F1F5F8", "#EEF7F4", "#FBF5EF")
  )

  nodes <- data.frame(
    id = c(
      "input", "config", "builder", "plan",
      "screen", "enhanced", "formal",
      "promotion", "registry", "resident", "streamed", "output"
    ),
    xmin = c(5, 44, 59, 76, 68, 41, 14, 6, 29, 54, 54, 78),
    xmax = c(18, 55, 72, 92, 86, 59, 32, 23, 47, 70, 70, 96),
    ymin = c(75, 75, 75, 75, 44, 44, 44, 15, 15, 21, 9, 15),
    ymax = c(86, 86, 86, 86, 54, 54, 54, 25, 25, 29, 17, 25),
    label = c(
      "Official shape\n+ device fingerprint",
      "ConfigSpec",
      "PlanBuilder\nstatic legality",
      "ExecutionPlan",
      "Screen",
      "Enhanced",
      "Formal\npaired blocks",
      "Sequential promotion\n>=2% + confidence",
      "Exact-device registry\nGPU + shape",
      "Shapes 01-13\nresident runtime",
      "Shape 14\nstreamed runtime",
      "Official-compatible\nTransformer output"
    ),
    fill = c(
      "#FFFFFF", "#E6EFF6", "#E6EFF6", "#E6EFF6",
      "#DCEFEA", "#DCEFEA", "#DCEFEA",
      "#F8E8DA", "#F8E8DA", "#E4F2EE", "#F8E8DA", "#FFFFFF"
    ),
    border = c(
      REPORT_COLORS[["secondary"]],
      rep(REPORT_COLORS[["navy"]], 3),
      rep(REPORT_COLORS[["teal"]], 3),
      rep(REPORT_COLORS[["orange"]], 2),
      REPORT_COLORS[["teal"]], REPORT_COLORS[["orange"]],
      REPORT_COLORS[["secondary"]]
    ),
    stringsAsFactors = FALSE
  )

  search_chips <- data.frame(
    xmin = c(22, 22),
    xmax = c(40, 40),
    ymin = c(77.0, 70.0),
    ymax = c(82.5, 75.5),
    label = c("Resident TPE\nShapes 01-13", "Finite streamed search\nShape 14"),
    fill = c(REPORT_COLORS[["navy"]], REPORT_COLORS[["orange"]]),
    stringsAsFactors = FALSE
  )

  p <- ggplot() +
    geom_rect(
      data = bands,
      aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = fill),
      colour = NA,
      show.legend = FALSE
    ) +
    scale_fill_identity() +
    geom_text(
      data = data.frame(
        x = c(4, 4, 4),
        y = c(93.5, 58.5, 28.5),
        label = c(
          "1  PROGRAM SYNTHESIS & STATIC COMPILATION",
          "2  ISOLATED MULTI-FIDELITY EVIDENCE",
          "3  PROMOTION, DEPLOYMENT & EXECUTION"
        )
      ),
      aes(x = x, y = y, label = label),
      hjust = 0,
      vjust = 0.5,
      family = "Arial",
      fontface = "bold",
      size = 8.2 / .architecture_pt,
      colour = REPORT_COLORS[["ink"]]
    ) +
    annotate(
      "rect",
      xmin = 20,
      xmax = 42,
      ymin = 68,
      ymax = 88,
      fill = "#FFFFFF",
      colour = REPORT_COLORS[["navy"]],
      linewidth = 0.5
    ) +
    annotate(
      "text",
      x = 22,
      y = 85.4,
      label = "Conditional search space",
      hjust = 0,
      family = "Arial",
      fontface = "bold",
      size = 7.7 / .architecture_pt,
      colour = REPORT_COLORS[["ink"]]
    ) +
    geom_rect(
      data = search_chips,
      aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = fill),
      colour = NA,
      show.legend = FALSE
    ) +
    geom_text(
      data = search_chips,
      aes(x = (xmin + xmax) / 2, y = (ymin + ymax) / 2, label = label),
      family = "Arial",
      fontface = "bold",
      size = 6.8 / .architecture_pt,
      lineheight = 0.92,
      colour = "white"
    ) +
    geom_rect(
      data = nodes,
      aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
      fill = nodes$fill,
      colour = nodes$border,
      linewidth = 0.5
    ) +
    geom_text(
      data = nodes,
      aes(x = (xmin + xmax) / 2, y = (ymin + ymax) / 2, label = label),
      family = "Arial",
      fontface = ifelse(
        nodes$id %in% c("config", "plan", "screen", "enhanced"),
        "bold",
        "plain"
      ),
      size = 7.2 / .architecture_pt,
      lineheight = 0.94,
      colour = REPORT_COLORS[["ink"]]
    ) +
    annotate(
      "segment", x = 18, xend = 20, y = 80.5, yend = 80.5,
      linewidth = 0.52, colour = "#89939A",
      arrow = .architecture_arrow()
    ) +
    annotate(
      "segment", x = 42, xend = 44, y = 80.5, yend = 80.5,
      linewidth = 0.52, colour = "#89939A",
      arrow = .architecture_arrow()
    ) +
    annotate(
      "segment", x = 55, xend = 59, y = 80.5, yend = 80.5,
      linewidth = 0.52, colour = "#89939A",
      arrow = .architecture_arrow()
    ) +
    annotate(
      "segment", x = 72, xend = 76, y = 80.5, yend = 80.5,
      linewidth = 0.52, colour = "#89939A",
      arrow = .architecture_arrow()
    ) +
    annotate(
      "text",
      x = 66,
      y = 69.0,
      label = "Invalid combinations are rejected before GPU execution",
      family = "Arial",
      size = 6.8 / .architecture_pt,
      colour = REPORT_COLORS[["secondary"]]
    ) +
    annotate(
      "segment", x = 84, xend = 84, y = 75, yend = 64,
      linewidth = 0.52, colour = "#89939A"
    ) +
    annotate(
      "segment", x = 84, xend = 77, y = 64, yend = 64,
      linewidth = 0.52, colour = "#89939A"
    ) +
    annotate(
      "segment", x = 77, xend = 77, y = 64, yend = 54,
      linewidth = 0.52, colour = "#89939A",
      arrow = .architecture_arrow()
    ) +
    annotate(
      "segment", x = 68, xend = 59, y = 49, yend = 49,
      linewidth = 0.52, colour = "#89939A",
      arrow = .architecture_arrow()
    ) +
    annotate(
      "segment", x = 41, xend = 32, y = 49, yend = 49,
      linewidth = 0.52, colour = "#89939A",
      arrow = .architecture_arrow()
    ) +
    annotate(
      "curve",
      x = 50,
      xend = 31,
      y = 54,
      yend = 68,
      curvature = -0.23,
      linewidth = 0.5,
      colour = REPORT_COLORS[["teal"]],
      arrow = .architecture_arrow(REPORT_COLORS[["teal"]], 1.45)
    ) +
    annotate(
      "text",
      x = 54,
      y = 59.4,
      label = "latency, failures, and coverage update the sampler",
      family = "Arial",
      size = 6.8 / .architecture_pt,
      colour = REPORT_COLORS[["teal"]]
    ) +
    annotate(
      "text",
      x = 50,
      y = 39.2,
      label = "single-GPU lease  |  fresh process  |  accuracy + path + latency + memory",
      family = "Arial",
      size = 6.8 / .architecture_pt,
      colour = REPORT_COLORS[["secondary"]]
    ) +
    annotate(
      "segment", x = 23, xend = 0.8, y = 44, yend = 35,
      linewidth = 0.52, colour = REPORT_COLORS[["orange"]]
    ) +
    annotate(
      "segment", x = 0.8, xend = 0.8, y = 35, yend = 20,
      linewidth = 0.52, colour = REPORT_COLORS[["orange"]]
    ) +
    annotate(
      "segment", x = 0.8, xend = 4, y = 20, yend = 20,
      linewidth = 0.52, colour = REPORT_COLORS[["orange"]],
      arrow = .architecture_arrow(REPORT_COLORS[["orange"]], 1.45)
    ) +
    annotate(
      "segment", x = 23, xend = 29, y = 20, yend = 20,
      linewidth = 0.52, colour = REPORT_COLORS[["orange"]],
      arrow = .architecture_arrow(REPORT_COLORS[["orange"]], 1.45)
    ) +
    annotate(
      "text",
      x = 26,
      y = 22.1,
      label = "winner",
      family = "Arial",
      size = 6.6 / .architecture_pt,
      colour = REPORT_COLORS[["orange"]]
    ) +
    annotate(
      "segment", x = 47, xend = 50.5, y = 20, yend = 20,
      linewidth = 0.52, colour = "#89939A"
    ) +
    annotate(
      "segment", x = 50.5, xend = 50.5, y = 20, yend = 25,
      linewidth = 0.52, colour = "#89939A"
    ) +
    annotate(
      "segment", x = 50.5, xend = 54, y = 25, yend = 25,
      linewidth = 0.52, colour = REPORT_COLORS[["teal"]],
      arrow = .architecture_arrow(REPORT_COLORS[["teal"]], 1.45)
    ) +
    annotate(
      "segment", x = 50.5, xend = 50.5, y = 20, yend = 13,
      linewidth = 0.52, colour = "#89939A"
    ) +
    annotate(
      "segment", x = 50.5, xend = 54, y = 13, yend = 13,
      linewidth = 0.52, colour = REPORT_COLORS[["orange"]],
      arrow = .architecture_arrow(REPORT_COLORS[["orange"]], 1.45)
    ) +
    annotate(
      "segment", x = 70, xend = 78, y = 25, yend = 21,
      linewidth = 0.52, colour = REPORT_COLORS[["teal"]],
      arrow = .architecture_arrow(REPORT_COLORS[["teal"]], 1.45)
    ) +
    annotate(
      "segment", x = 70, xend = 78, y = 13, yend = 19,
      linewidth = 0.52, colour = REPORT_COLORS[["orange"]],
      arrow = .architecture_arrow(REPORT_COLORS[["orange"]], 1.45)
    ) +
    coord_cartesian(xlim = c(0, 100), ylim = c(0, 100), clip = "off", expand = FALSE) +
    theme_report_void(base_size = 9, base_family = "Arial") +
    theme(plot.margin = margin(4, 5, 4, 5, unit = "mm"))

  p
}
