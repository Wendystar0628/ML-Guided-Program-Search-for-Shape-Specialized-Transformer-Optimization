## Workload-regime and sensitivity figures for the technical report.
## This file returns plot objects only; export is owned by the report renderer.

suppressPackageStartupMessages({
  library(ggplot2)
  library(ggrepel)
})

.workload_script_dir <- local({
  source_file <- ""
  frames <- sys.frames()
  for (index in rev(seq_along(frames))) {
    candidate <- frames[[index]]$ofile
    if (!is.null(candidate) && nzchar(candidate)) {
      source_file <- candidate
      break
    }
  }
  if (nzchar(source_file)) {
    if (!file.exists(source_file) && file.exists(basename(source_file))) {
      source_file <- basename(source_file)
    }
    dirname(normalizePath(source_file, winslash = "/", mustWork = FALSE))
  } else {
    normalizePath(getwd(), winslash = "/", mustWork = FALSE)
  }
})

if (!exists("theme_report", mode = "function") || !exists("REPORT_COLORS")) {
  source(file.path(.workload_script_dir, "theme_report.R"), local = FALSE)
}

.workload_source_dir <- dirname(report_source_data_path("workloads.csv"))

.read_workload_data <- function() {
  workload_path <- file.path(.workload_source_dir, "workloads.csv")
  performance_path <- file.path(.workload_source_dir, "performance.csv")

  workloads <- read.csv(
    workload_path,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  performance <- read.csv(
    performance_path,
    stringsAsFactors = FALSE,
    check.names = FALSE,
    na.strings = c("", "NA")
  )

  workload_fields <- c(
    "case_id", "batch_size", "qkv_dim", "heads", "seq_len", "total_tokens"
  )
  performance_fields <- c("case_id", "speedup")
  missing_workload <- setdiff(workload_fields, names(workloads))
  missing_performance <- setdiff(performance_fields, names(performance))
  if (length(missing_workload) > 0L || length(missing_performance) > 0L) {
    stop(
      "Missing required workload figure fields: ",
      paste(c(missing_workload, missing_performance), collapse = ", ")
    )
  }
  if (anyDuplicated(workloads$case_id) || anyDuplicated(performance$case_id)) {
    stop("case_id must be unique in both workload figure source tables")
  }

  rows <- merge(
    workloads,
    performance[, performance_fields],
    by = "case_id",
    all.x = TRUE,
    sort = FALSE
  )
  rows <- rows[match(workloads$case_id, rows$case_id), ]
  rows$shape <- sub("^official_", "", rows$case_id)
  rows$is_shape14 <- rows$case_id == "official_14"
  rows$execution_regime <- factor(
    ifelse(rows$is_shape14, "Shape 14 streamed", "Shapes 01-13"),
    levels = c("Shapes 01-13", "Shape 14 streamed")
  )
  rows$head_count <- factor(
    rows$heads,
    levels = c(1, 2, 4, 16),
    labels = c("1", "2", "4", "16")
  )
  rows$plot_label <- rows$shape
  shared_regime <- rows$seq_len == 128 & rows$total_tokens == 8192
  rows$plot_label[shared_regime] <- NA_character_

  if (any(rows$seq_len <= 0) || any(rows$total_tokens <= 0)) {
    stop("Log-scale workload coordinates must be strictly positive")
  }
  rows
}

.make_sensitivity_data <- function(rows) {
  sweep_specs <- list(
    list(
      title = "Batch size",
      ids = c("official_02", "official_03", "official_04", "official_01", "official_05", "official_06"),
      labels = c("1", "4", "16", "64", "128", "10,000")
    ),
    list(
      title = "QKV width",
      ids = c("official_07", "official_01", "official_08"),
      labels = c("32", "128", "1,024")
    ),
    list(
      title = "Head count",
      ids = c("official_09", "official_10", "official_01", "official_11"),
      labels = c("1", "2", "4", "16")
    ),
    list(
      title = "Sequence length",
      ids = c("official_12", "official_01", "official_13"),
      labels = c("32", "128", "1,024")
    )
  )

  panels <- lapply(seq_along(sweep_specs), function(sweep_index) {
    spec <- sweep_specs[[sweep_index]]
    selected <- rows[match(spec$ids, rows$case_id), ]
    if (any(is.na(selected$case_id)) || any(is.na(selected$speedup))) {
      stop("Sensitivity source rows or speedup values are missing for ", spec$title)
    }
    selected$sweep <- spec$title
    selected$x_label <- spec$labels
    selected$x_order <- seq_along(spec$ids)
    selected$x_key <- sprintf("%02d__%02d__%s", sweep_index, selected$x_order, selected$x_label)
    selected
  })

  sensitivity <- do.call(rbind, panels)
  sweep_levels <- vapply(
    sweep_specs,
    function(spec) spec$title,
    character(1)
  )
  sensitivity$sweep <- factor(sensitivity$sweep, levels = sweep_levels)
  sensitivity$x_key <- factor(sensitivity$x_key, levels = unique(sensitivity$x_key))
  sensitivity$is_reference <- sensitivity$case_id == "official_01"
  sensitivity$point_kind <- factor(
    ifelse(sensitivity$is_reference, "Shape 01 reference", "Other shapes"),
    levels = c("Other shapes", "Shape 01 reference")
  )
  sensitivity$is_extreme <- as.logical(ave(
    sensitivity$speedup,
    sensitivity$sweep,
    FUN = function(values) values == min(values) | values == max(values)
  ))
  sensitivity$extreme_label <- ifelse(
    sensitivity$is_extreme,
    sprintf("%.1fx", sensitivity$speedup),
    NA_character_
  )
  sensitivity
}

draw_workload_landscape <- function() {
  rows <- .read_workload_data()

  ggplot(
    rows,
    aes(
      x = seq_len,
      y = total_tokens,
      shape = head_count,
      size = qkv_dim,
      colour = execution_regime
    )
  ) +
    geom_point(alpha = 0.92, stroke = 0.55) +
    geom_text_repel(
      data = rows[!is.na(rows$plot_label), , drop = FALSE],
      aes(label = plot_label),
      seed = 20260831,
      size = 2.8,
      colour = REPORT_COLORS[["ink"]],
      segment.colour = REPORT_COLORS[["secondary"]],
      segment.size = 0.28,
      min.segment.length = 0,
      box.padding = 0.32,
      point.padding = 0.28,
      max.overlaps = Inf,
      show.legend = FALSE
    ) +
    annotate(
      "segment",
      x = 250,
      xend = 138,
      y = 5100,
      yend = 7600,
      colour = REPORT_COLORS[["secondary"]],
      linewidth = 0.28
    ) +
    annotate(
      "text",
      x = 275,
      y = 4600,
      label = "01, 07-11\nshared B and S",
      hjust = 0,
      vjust = 0.5,
      size = 2.7,
      lineheight = 0.9,
      colour = REPORT_COLORS[["ink"]]
    ) +
    scale_x_log10(
      breaks = c(32, 128, 1024, 100000),
      labels = c("32", "128", "1,024", "100,000"),
      expand = expansion(mult = c(0.10, 0.13))
    ) +
    scale_y_log10(
      breaks = c(100, 1000, 10000, 100000, 1000000),
      labels = c("100", "1k", "10k", "100k", "1M"),
      expand = expansion(mult = c(0.08, 0.12))
    ) +
    scale_shape_manual(
      name = "Attention heads",
      values = c("1" = 16, "2" = 15, "4" = 18, "16" = 17),
      drop = FALSE
    ) +
    scale_size_continuous(
      name = "QKV width",
      range = c(2.8, 7.2),
      breaks = c(32, 128, 1024),
      labels = c("32", "128", "1,024")
    ) +
    scale_colour_manual(
      name = NULL,
      values = c(
        "Shapes 01-13" = REPORT_COLORS[["navy"]],
        "Shape 14 streamed" = REPORT_COLORS[["orange"]]
      )
    ) +
    guides(
      colour = guide_legend(order = 1, override.aes = list(shape = 16, size = 3.4)),
      shape = guide_legend(order = 2, override.aes = list(size = 3.4)),
      size = guide_legend(order = 3, override.aes = list(shape = 16))
    ) +
    labs(
      title = "Official workload regime map",
      subtitle = "Head count is encoded by shape; QKV width is encoded by marker area.",
      x = "Sequence length S (log scale)",
      y = "Logical tokens B x S (log scale)"
    ) +
    theme_report(base_size = 9, base_family = "Arial") +
    theme(
      legend.position = "top",
      legend.box = "vertical",
      legend.title = element_text(
        size = 8,
        face = "bold",
        colour = REPORT_COLORS[["secondary"]]
      ),
      legend.margin = margin(0, 0, 2, 0),
      legend.key.width = grid::unit(10, "pt"),
      legend.spacing.x = grid::unit(3, "pt"),
      plot.margin = margin(8, 12, 7, 8),
      panel.grid.major = element_line(
        colour = REPORT_COLORS[["grid"]],
        linewidth = 0.35
      ),
      panel.grid.minor = element_blank()
    ) +
    coord_cartesian(clip = "off")
}

draw_workload_sensitivity <- function() {
  sensitivity <- .make_sensitivity_data(.read_workload_data())

  ggplot(
    sensitivity,
    aes(x = x_key, y = speedup, fill = point_kind, colour = point_kind)
  ) +
    geom_point(shape = 21, size = 3.2, stroke = 0.9) +
    geom_text_repel(
      data = sensitivity[sensitivity$is_extreme, ],
      aes(label = extreme_label),
      seed = 20260831,
      size = 2.7,
      colour = REPORT_COLORS[["ink"]],
      segment.colour = REPORT_COLORS[["secondary"]],
      segment.size = 0.25,
      min.segment.length = 0,
      box.padding = 0.28,
      point.padding = 0.25,
      nudge_y = 1.6,
      max.overlaps = Inf,
      show.legend = FALSE
    ) +
    facet_wrap(vars(sweep), ncol = 2, scales = "free_x") +
    scale_x_discrete(
      labels = function(keys) sub("^.*__", "", keys),
      expand = expansion(add = 0.55)
    ) +
    scale_y_continuous(
      limits = c(0, 40),
      breaks = seq(0, 40, 10),
      expand = expansion(mult = c(0, 0.04))
    ) +
    scale_fill_manual(
      name = NULL,
      values = c(
        "Other shapes" = REPORT_COLORS[["navy"]],
        "Shape 01 reference" = "white"
      )
    ) +
    scale_colour_manual(
      name = NULL,
      values = c(
        "Other shapes" = REPORT_COLORS[["navy"]],
        "Shape 01 reference" = REPORT_COLORS[["secondary"]]
      )
    ) +
    guides(
      fill = guide_legend(
        override.aes = list(
          shape = 21,
          size = 3.2,
          colour = c(REPORT_COLORS[["navy"]], REPORT_COLORS[["secondary"]])
        )
      ),
      colour = "none"
    ) +
    labs(
      title = "Shape sensitivity without implied interpolation",
      subtitle = "Each point is an independently deployed plan; extrema are labelled directly.",
      x = NULL,
      y = "Speedup (x)"
    ) +
    theme_report(base_size = 9, base_family = "Arial") +
    theme(
      legend.position = "top",
      legend.margin = margin(0, 0, 2, 0),
      panel.spacing = grid::unit(12, "pt"),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_line(
        colour = REPORT_COLORS[["grid"]],
        linewidth = 0.35
      ),
      strip.background = element_blank(),
      strip.text = element_text(
        colour = REPORT_COLORS[["ink"]],
        face = "bold",
        hjust = 0
      ),
      plot.margin = margin(8, 9, 7, 8)
    ) +
    coord_cartesian(clip = "off")
}
