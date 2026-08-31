# Deployed-program decision matrix for the technical report and README.
#
# The matrix intentionally summarizes complete plans along seven decision axes.
# Exact values remain traceable to source_data/deployed_programs.csv and the
# deployment registry; colour is redundant with direct cell labels.

.matrix_script_dir <- local({
  source_file <- tryCatch(sys.frame(1)$ofile, error = function(...) NULL)
  if (is.null(source_file) || !nzchar(source_file)) {
    file.path(getwd(), "R")
  } else {
    dirname(normalizePath(source_file, winslash = "/", mustWork = FALSE))
  }
})

if (!exists("theme_report", mode = "function") || !exists("REPORT_COLORS")) {
  source(file.path(.matrix_script_dir, "theme_report.R"), local = TRUE)
}

.matrix_pt <- 72.27 / 25.4

.matrix_shape_label <- function(shape_label) {
  parts <- strsplit(enc2utf8(shape_label), "\u00b7", fixed = TRUE)
  vapply(parts, function(piece) {
    piece <- trimws(piece)
    if (length(piece) < 2L) {
      return(piece[[1]])
    }
    dims <- trimws(piece[[2]])
    dims <- gsub("B10000", "B10k", dims, fixed = TRUE)
    dims <- gsub("S100000", "S100k", dims, fixed = TRUE)
    dims <- gsub(" / ", " ", dims, fixed = TRUE)
    dims <- sub(" D", "\nD", dims, fixed = TRUE)
    dims <- strsplit(dims, "\n", fixed = TRUE)[[1]]
    paste0(piece[[1]], "  ", dims[[1]], "\n", dims[[2]])
  }, character(1))
}

.matrix_schedule_label <- function(value) {
  value <- sub("^Compiled$", "torch.compile", value)
  value <- sub("^Tile graph 128$", "Tile graph\n128", value)
  value <- sub("^Streamed mb2$", "Streamed\nmicrobatch 2", value)
  value
}

.matrix_attention_label <- function(value) {
  replacements <- c(
    "Efficient SDPA" = "Efficient\nSDPA",
    "Causal SDPA" = "Causal\nSDPA",
    "cuDNN SDPA" = "cuDNN\nSDPA",
    "Triton Dh8" = "Triton\nhead dim 8",
    "Triton S1024" = "Triton\nS = 1024",
    "Triton stream Dh64" = "Triton stream\nhead dim 64"
  )
  unname(ifelse(value %in% names(replacements), replacements[value], value))
}

.matrix_layout_label <- function(value) {
  value <- gsub(intToUtf8(0x2192), "to", value, fixed = TRUE)
  replacements <- c(
    "Native to Triton O" = "Native to\nTriton output",
    "Native to Torch O" = "Native to\nTorch output",
    "View to Torch O" = "View to\nTorch output",
    "Native to direct BSD" = "Native to\ndirect BSD",
    "View to direct BSD" = "View to\ndirect BSD"
  )
  unname(ifelse(value %in% names(replacements), replacements[value], value))
}

.matrix_projection_label <- function(value) {
  paste(strsplit(value, "", fixed = TRUE)[[1]], collapse = " / ")
}

.matrix_ffn_label <- function(value) {
  replacements <- c(
    "Fused boundary" = "Fused\nboundary",
    "Linear + GELU" = "Linear +\nGELU"
  )
  unname(ifelse(value %in% names(replacements), replacements[value], value))
}

.matrix_norm_label <- function(value) {
  replacements <- c(
    "T16 / Lin-mixed" = "Triton-16\nLinear mixed",
    "T16 / T-mixed" = "Triton-16\nTriton mixed",
    "Torch / T-mixed" = "Torch\nTriton mixed",
    "Torch / Torch" = "Torch",
    "Fused QKV / Lin-mixed" = "Fused QKV\nLinear mixed"
  )
  unname(ifelse(value %in% names(replacements), replacements[value], value))
}

.matrix_precision_label <- function(value) {
  replacements <- c(
    "FP16 core" = "FP16\ncore",
    "Attn + FFN-in FP16" = "Attention +\nFFN-in FP16"
  )
  unname(ifelse(value %in% names(replacements), replacements[value], value))
}

.matrix_component_state <- function(axis, raw_value, case_id) {
  streamed <- case_id == "official_14" && axis %in% c("Runtime", "Attention", "Precision")
  optimized <- switch(
    axis,
    Runtime = raw_value != "Eager",
    Attention = grepl("Triton", raw_value, fixed = TRUE),
    Output = grepl("Triton|direct BSD", raw_value),
    Projection = grepl("S", raw_value, fixed = TRUE),
    FFN = raw_value != "Torch",
    `Norm / fusion` = raw_value != "Torch / Torch",
    Precision = raw_value != "FP16 core",
    FALSE
  )
  ifelse(streamed, "streamed", ifelse(optimized, "optimized", "library"))
}

.matrix_long_data <- function(rows) {
  axes <- c(
    "Runtime", "Attention", "Output", "Projection", "FFN",
    "Norm / fusion", "Precision"
  )
  raw <- list(
    rows$schedule,
    rows$attention,
    rows$layout_bridge,
    rows$projections,
    rows$ffn,
    rows$norms,
    rows$precision
  )
  labels <- list(
    vapply(rows$schedule, .matrix_schedule_label, character(1)),
    vapply(rows$attention, .matrix_attention_label, character(1)),
    vapply(rows$layout_bridge, .matrix_layout_label, character(1)),
    vapply(rows$projections, .matrix_projection_label, character(1)),
    vapply(rows$ffn, .matrix_ffn_label, character(1)),
    vapply(rows$norms, .matrix_norm_label, character(1)),
    vapply(rows$precision, .matrix_precision_label, character(1))
  )

  cells <- do.call(
    rbind,
    lapply(seq_along(axes), function(index) {
      data.frame(
        case_id = rows$case_id,
        y = rows$y,
        axis = axes[[index]],
        raw_value = raw[[index]],
        label = labels[[index]],
        stringsAsFactors = FALSE
      )
    })
  )
  cells$axis <- factor(cells$axis, levels = axes)
  cells$state <- factor(
    mapply(
      .matrix_component_state,
      as.character(cells$axis),
      cells$raw_value,
      cells$case_id,
      USE.NAMES = FALSE
    ),
    levels = c("library", "optimized", "streamed")
  )
  cells
}

#' Draw the exact-device deployed-program decision matrix.
#'
#' @param data_path CSV produced by the report data-preparation step.
#' @return A ggplot object. Rendering/export is left to the shared R driver.
draw_deployed_program_matrix <- function(
    data_path = file.path("source_data", "deployed_programs.csv")) {
  rows <- utils::read.csv(
    data_path,
    stringsAsFactors = FALSE,
    check.names = FALSE,
    encoding = "UTF-8"
  )
  required <- c(
    "case_id", "shape_label", "schedule", "attention", "layout_bridge",
    "projections", "ffn", "norms", "precision"
  )
  missing <- setdiff(required, names(rows))
  if (length(missing) > 0L) {
    stop("deployed_programs.csv is missing: ", paste(missing, collapse = ", "))
  }

  rows <- rows[order(rows$case_id), , drop = FALSE]
  if (!identical(rows$case_id, sprintf("official_%02d", 1:14))) {
    stop("Expected exactly one ordered deployment row for each official Shape 01-14.")
  }

  rows$y <- rev(seq_len(nrow(rows)))
  rows$shape_display <- .matrix_shape_label(rows$shape_label)
  cells <- .matrix_long_data(rows)
  axis_levels <- c(
    "Shape and\ndimensions", "Runtime", "Attention", "Output", "Projection",
    "FFN", "Norm / fusion", "Precision"
  )
  cells$axis <- factor(as.character(cells$axis), levels = axis_levels)
  shape_cells <- data.frame(
    case_id = rows$case_id,
    y = rows$y,
    axis = factor("Shape and\ndimensions", levels = axis_levels),
    raw_value = rows$shape_label,
    label = rows$shape_display,
    state = factor("shape", levels = c("shape", "library", "optimized", "streamed")),
    stringsAsFactors = FALSE
  )
  cells$state <- factor(
    as.character(cells$state),
    levels = c("shape", "library", "optimized", "streamed")
  )
  cells <- rbind(shape_cells, cells)

  state_colours <- c(
    shape = "#FFFFFF",
    library = "#E8EDF0",
    optimized = "#D9ECE7",
    streamed = "#F7E5D5"
  )
  state_labels <- c(
    library = "Library / native",
    optimized = "Optimized / specialized",
    streamed = "Shape 14 streamed"
  )

  ggplot2::ggplot() +
    ggplot2::geom_rect(
      data = rows[rows$case_id == "official_14", , drop = FALSE],
      ggplot2::aes(xmin = 0.50, xmax = 8.50, ymin = y - 0.48, ymax = y + 0.48),
      fill = "#FFF7F0",
      colour = REPORT_COLORS[["orange"]],
      linewidth = 0.55,
      inherit.aes = FALSE
    ) +
    ggplot2::geom_tile(
      data = cells,
      ggplot2::aes(x = axis, y = y, fill = state),
      width = 0.92,
      height = 0.78,
      colour = "white",
      linewidth = 0.45
    ) +
    ggplot2::geom_text(
      data = cells[cells$axis != "Shape and\ndimensions", , drop = FALSE],
      ggplot2::aes(x = axis, y = y, label = label),
      family = "Arial",
      colour = REPORT_COLORS[["ink"]],
      size = 6.4 / .matrix_pt,
      lineheight = 0.92
    ) +
    ggplot2::geom_text(
      data = cells[cells$axis == "Shape and\ndimensions", , drop = FALSE],
      ggplot2::aes(x = axis, y = y, label = label),
      vjust = 0.5,
      family = "Arial",
      fontface = "bold",
      colour = REPORT_COLORS[["ink"]],
      size = 6.8 / .matrix_pt,
      lineheight = 0.94
    ) +
    ggplot2::geom_segment(
      data = rows,
      ggplot2::aes(x = 0.50, xend = 8.50, y = y - 0.50, yend = y - 0.50),
      colour = REPORT_COLORS[["grid"]],
      linewidth = 0.30,
      inherit.aes = FALSE
    ) +
    ggplot2::scale_fill_manual(
      values = state_colours,
      breaks = c("library", "optimized", "streamed"),
      labels = state_labels,
      drop = FALSE
    ) +
    ggplot2::scale_x_discrete(
      position = "top",
      expand = ggplot2::expansion(add = 0.50)
    ) +
    ggplot2::scale_y_continuous(
      breaks = NULL,
      limits = c(0.45, 14.85),
      expand = c(0, 0)
    ) +
    ggplot2::coord_cartesian(clip = "off") +
    ggplot2::labs(
      fill = NULL,
      caption = paste0(
        "Projection order is Q / K / V / O: S = FP16 shadow, A = autocast FP16, ",
        "I = input dtype. Full typed configurations remain in the deployment registry."
      )
    ) +
    ggplot2::guides(
      fill = ggplot2::guide_legend(
        direction = "horizontal",
        title.position = "top",
        keywidth = grid::unit(4.0, "mm"),
        keyheight = grid::unit(3.0, "mm"),
        nrow = 1,
        byrow = TRUE
      )
    ) +
    ggplot2::theme_void(base_family = "Arial", base_size = 8) +
    ggplot2::theme(
      text = ggplot2::element_text(colour = REPORT_COLORS[["ink"]]),
      axis.text.x = ggplot2::element_text(
        family = "Arial",
        face = "bold",
        colour = REPORT_COLORS[["secondary"]],
        size = 6.4,
        margin = ggplot2::margin(b = 4)
      ),
      legend.position = "top",
      legend.justification = "left",
      legend.text = ggplot2::element_text(
        family = "Arial",
        colour = REPORT_COLORS[["secondary"]],
        size = 6.4
      ),
      legend.margin = ggplot2::margin(0, 0, 1, 0),
      legend.box.spacing = grid::unit(1.2, "mm"),
      plot.caption = ggplot2::element_text(
        family = "Arial",
        colour = REPORT_COLORS[["secondary"]],
        size = 6.0,
        hjust = 0,
        margin = ggplot2::margin(t = 4)
      ),
      plot.margin = ggplot2::margin(3, 4, 3, 4, unit = "mm")
    )
}
