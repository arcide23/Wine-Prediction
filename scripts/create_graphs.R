#!/usr/bin/env Rscript

project_root <- getwd()
output_dir <- file.path(project_root, "output")
summary_file <- file.path(output_dir, "cv_model_summary.csv")
fold_file <- file.path(output_dir, "cv_fold_results.csv")

if (!file.exists(summary_file)) {
  stop("Summary file not found at ", summary_file, ". Run scripts/train_models.R first.")
}

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
}

summary_df <- read.csv(summary_file, header = TRUE, stringsAsFactors = FALSE)

model_order <- c(
  "intercept",
  "all_parameters",
  "all_plus_squared",
  "all_plus_squared_interactions",
  "all_plus_squared_plus_interactions_plus_squared_interactions",
  "lasso",
  "ridge"
)
summary_df <- summary_df[match(model_order, summary_df$model), , drop = FALSE]
summary_df <- summary_df[!is.na(summary_df$model), , drop = FALSE]

pdf(file.path(output_dir, "avg_mse_comparison.pdf"), width = 8, height = 5)
barplot(
  height = summary_df$avg_mse,
  names.arg = summary_df$model,
  col = c("#7f7f7f", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#377eb8", "#e41a1c"),
  ylab = "Average MSE",
  xlab = "Model",
  main = "5-Fold CV Average MSE by Model",
  las = 2
)
dev.off()

if (file.exists(fold_file)) {
  fold_df <- read.csv(fold_file, header = TRUE, stringsAsFactors = FALSE)
  fold_df$model <- factor(fold_df$model, levels = model_order)
  fold_df <- fold_df[!is.na(fold_df$model), , drop = FALSE]

  pdf(file.path(output_dir, "fold_mse_distribution.pdf"), width = 8, height = 5)
  boxplot(
    mse ~ model,
    data = fold_df,
    col = c("#7f7f7f", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#377eb8", "#e41a1c"),
    ylab = "Fold MSE",
    xlab = "Model",
    main = "5-Fold CV MSE Distribution by Model",
    las = 2
  )
  dev.off()
}

message("Wrote PDF plots to: ", output_dir)
