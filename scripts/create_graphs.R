#!/usr/bin/env Rscript

project_root <- getwd()
output_dir <- file.path(project_root, "output")
fold_file <- file.path(output_dir, "cv_fold_results.csv")

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
}

model_order <- c(
  "intercept",
  "all_parameters",
  "all_plus_squared",
  "all_plus_squared_interactions",
  "all_plus_squared_plus_interactions_plus_squared_interactions",
  "lasso",
  "ridge",
  "lasso_all_plus_squared",
  "ridge_all_plus_squared"
)

if (!file.exists(fold_file)) {
  stop("Fold results file not found at ", fold_file, ". Run scripts/train_models.R first.")
}

fold_df <- read.csv(fold_file, header = TRUE, stringsAsFactors = FALSE)
fold_df <- fold_df[fold_df$model %in% model_order, , drop = FALSE]

# Remove outliers per model based on the 1.5 * IQR rule.
keep_rows <- rep(TRUE, nrow(fold_df))
for (model_name in unique(fold_df$model)) {
  idx <- which(fold_df$model == model_name)
  vals <- fold_df$accuracy[idx]
  q1 <- as.numeric(quantile(vals, 0.25, na.rm = TRUE, names = FALSE))
  q3 <- as.numeric(quantile(vals, 0.75, na.rm = TRUE, names = FALSE))
  iqr <- q3 - q1
  lower <- q1 - 1.5 * iqr
  upper <- q3 + 1.5 * iqr
  keep_rows[idx] <- vals >= lower & vals <= upper
}

fold_df_filtered <- fold_df[keep_rows, , drop = FALSE]

summary_df <- aggregate(accuracy ~ model, data = fold_df_filtered, FUN = mean)
names(summary_df)[names(summary_df) == "accuracy"] <- "avg_accuracy"
summary_df$model <- factor(summary_df$model, levels = model_order)
summary_df <- summary_df[order(summary_df$model), , drop = FALSE]
summary_df <- summary_df[!is.na(summary_df$model), , drop = FALSE]

pdf(file.path(output_dir, "avg_accuracy_comparison.pdf"), width = 8, height = 5)
barplot(
  height = summary_df$avg_accuracy,
  names.arg = summary_df$model,
  col = c("#7f7f7f", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#377eb8", "#e41a1c", "#17becf", "#bcbd22"),
  ylab = "Average Accuracy",
  xlab = "Model",
  main = "5-Fold CV Average Accuracy by Model",
  las = 2
)
dev.off()

fold_df_filtered$model <- factor(fold_df_filtered$model, levels = model_order)
fold_df_filtered <- fold_df_filtered[!is.na(fold_df_filtered$model), , drop = FALSE]

pdf(file.path(output_dir, "fold_accuracy_distribution.pdf"), width = 8, height = 5)
boxplot(
  accuracy ~ model,
  data = fold_df_filtered,
  col = c("#7f7f7f", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#377eb8", "#e41a1c", "#17becf", "#bcbd22"),
  ylab = "Fold Accuracy",
  xlab = "Model",
  main = "5-Fold CV Accuracy by Model (Outliers Removed)",
  las = 2
)
dev.off()

message("Wrote PDF plots to: ", output_dir)
message("Removed ", sum(!keep_rows), " outlier row(s) before plotting.")
