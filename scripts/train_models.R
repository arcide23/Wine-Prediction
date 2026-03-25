#!/usr/bin/env Rscript

suppressWarnings(suppressMessages({
  library(glmnet)
  library(nnet)
  if (!requireNamespace("randomForest", quietly = TRUE)) {
    stop(
      "Package 'randomForest' is required but not installed.\n",
      "Install it with:\n",
      "  install.packages('randomForest')\n"
    )
  }
  library(randomForest)
}))

project_root <- getwd()
input_file <- file.path(project_root, "data", "processed", "train.csv")
output_dir <- file.path(project_root, "output")

if (!file.exists(input_file)) {
  stop("Training file not found at ", input_file)
}

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
}

seed <- 2026
outer_folds <- 5
inner_folds <- 5

set.seed(seed)
train_df <- read.csv(input_file, header = TRUE, stringsAsFactors = FALSE)

if (!"quality" %in% names(train_df)) {
  stop("Column 'quality' not found in training data.")
}

train_df$quality <- factor(train_df$quality)
class_levels <- levels(train_df$quality)

n <- nrow(train_df)
fold_ids <- sample(rep(seq_len(outer_folds), length.out = n))

feature_cols <- setdiff(names(train_df), "quality")
x_all <- as.matrix(train_df[, feature_cols, drop = FALSE])
y_all <- train_df$quality

squared_terms <- paste0("I(", feature_cols, "^2)")
squared_formula <- as.formula(
  paste("quality ~", paste(c(feature_cols, squared_terms), collapse = " + "))
)
squared_interactions_formula <- as.formula(
  paste("quality ~ (", paste(feature_cols, collapse = " + "), ")^2 +", paste(squared_terms, collapse = " + "))
)
interaction_pairs <- combn(feature_cols, 2, simplify = FALSE)
squared_interaction_terms <- vapply(
  interaction_pairs,
  function(pair) paste0("I((", pair[1], " * ", pair[2], ")^2)"),
  character(1)
)
full_squared_interactions_formula <- as.formula(
  paste(
    "quality ~",
    paste(c(feature_cols, squared_terms, paste0("(", paste(feature_cols, collapse = " + "), ")^2"), squared_interaction_terms), collapse = " + ")
  )
)

results <- data.frame(
  fold = integer(0),
  model = character(0),
  accuracy = numeric(0),
  tuned_lambda = numeric(0),
  stringsAsFactors = FALSE
)

get_accuracy <- function(truth, pred) {
  mean(as.character(truth) == as.character(pred))
}

add_squared_columns <- function(df, feature_names) {
  out <- df
  for (nm in feature_names) {
    out[[paste0(nm, "_sq")]] <- out[[nm]]^2
  }
  out
}

fit_predict_multinom <- function(formula_obj, train_data, valid_data) {
  fit <- multinom(
    formula_obj,
    data = train_data,
    trace = FALSE,
    MaxNWts = 200000
  )
  pred <- predict(fit, newdata = valid_data, type = "class")
  factor(pred, levels = class_levels)
}

make_stratified_foldid <- function(y, k) {
  foldid <- integer(length(y))
  y_chr <- as.character(y)
  for (cls in unique(y_chr)) {
    cls_idx <- which(y_chr == cls)
    cls_idx <- sample(cls_idx)
    foldid[cls_idx] <- rep(seq_len(k), length.out = length(cls_idx))
  }
  foldid
}

for (fold in seq_len(outer_folds)) {
  valid_idx <- which(fold_ids == fold)
  train_idx <- which(fold_ids != fold)

  x_train <- x_all[train_idx, , drop = FALSE]
  y_train <- y_all[train_idx]
  x_valid <- x_all[valid_idx, , drop = FALSE]
  y_valid <- y_all[valid_idx]
  x_train_sq <- x_train^2
  x_valid_sq <- x_valid^2
  colnames(x_train_sq) <- paste0(colnames(x_train), "_sq")
  colnames(x_valid_sq) <- paste0(colnames(x_valid), "_sq")
  x_train_plus_sq <- cbind(x_train, x_train_sq)
  x_valid_plus_sq <- cbind(x_valid, x_valid_sq)

  # Intercept-only baseline predicts majority class in fold-training split.
  majority_class <- names(which.max(table(y_train)))
  pred_intercept <- factor(rep(majority_class, length(valid_idx)), levels = class_levels)
  acc_intercept <- get_accuracy(y_valid, pred_intercept)
  results <- rbind(
    results,
    data.frame(fold = fold, model = "intercept", accuracy = acc_intercept, tuned_lambda = NA_real_)
  )

  train_fold_df <- train_df[train_idx, c(feature_cols, "quality"), drop = FALSE]
  valid_fold_df <- train_df[valid_idx, c(feature_cols, "quality"), drop = FALSE]

  pred_all <- fit_predict_multinom(quality ~ ., train_fold_df, valid_fold_df)
  acc_all <- get_accuracy(y_valid, pred_all)
  results <- rbind(
    results,
    data.frame(fold = fold, model = "all_parameters", accuracy = acc_all, tuned_lambda = NA_real_)
  )

  pred_squared <- fit_predict_multinom(squared_formula, train_fold_df, valid_fold_df)
  acc_squared <- get_accuracy(y_valid, pred_squared)
  results <- rbind(
    results,
    data.frame(
      fold = fold,
      model = "all_plus_squared",
      accuracy = acc_squared,
      tuned_lambda = NA_real_
    )
  )

  pred_squared_interactions <- fit_predict_multinom(
    squared_interactions_formula,
    train_fold_df,
    valid_fold_df
  )
  acc_squared_interactions <- get_accuracy(y_valid, pred_squared_interactions)
  results <- rbind(
    results,
    data.frame(
      fold = fold,
      model = "all_plus_squared_interactions",
      accuracy = acc_squared_interactions,
      tuned_lambda = NA_real_
    )
  )

  pred_full_squared_interactions <- fit_predict_multinom(
    full_squared_interactions_formula,
    train_fold_df,
    valid_fold_df
  )
  acc_full_squared_interactions <- get_accuracy(y_valid, pred_full_squared_interactions)
  results <- rbind(
    results,
    data.frame(
      fold = fold,
      model = "all_plus_squared_plus_interactions_plus_squared_interactions",
      accuracy = acc_full_squared_interactions,
      tuned_lambda = NA_real_
    )
  )

  set.seed(seed + 400 + fold)
  rf_fit <- randomForest(quality ~ ., data = train_fold_df)
  rf_pred <- predict(rf_fit, newdata = valid_fold_df, type = "class")
  rf_pred <- factor(rf_pred, levels = class_levels)
  acc_rf <- get_accuracy(y_valid, rf_pred)
  results <- rbind(
    results,
    data.frame(
      fold = fold,
      model = "random_forest",
      accuracy = acc_rf,
      tuned_lambda = NA_real_
    )
  )

  set.seed(seed + 500 + fold)
  train_fold_sq <- add_squared_columns(train_fold_df, feature_cols)
  valid_fold_sq <- add_squared_columns(valid_fold_df, feature_cols)
  rf_sq_fit <- randomForest(quality ~ ., data = train_fold_sq)
  rf_sq_pred <- predict(rf_sq_fit, newdata = valid_fold_sq, type = "class")
  rf_sq_pred <- factor(rf_sq_pred, levels = class_levels)
  acc_rf_sq <- get_accuracy(y_valid, rf_sq_pred)
  results <- rbind(
    results,
    data.frame(
      fold = fold,
      model = "random_forest_all_plus_squared",
      accuracy = acc_rf_sq,
      tuned_lambda = NA_real_
    )
  )

  min_class_count <- min(table(y_train))
  inner_folds_current <- min(inner_folds, as.integer(min_class_count))
  while (inner_folds_current > 2 &&
    (min_class_count - ceiling(min_class_count / inner_folds_current)) < 2) {
    inner_folds_current <- inner_folds_current - 1
  }
  if (inner_folds_current < 2) {
    stop("Not enough observations per class to tune lasso/ridge with CV in this fold.")
  }
  inner_foldid <- make_stratified_foldid(y_train, inner_folds_current)

  set.seed(seed + fold)
  cv_lasso <- cv.glmnet(
    x = x_train,
    y = y_train,
    family = "multinomial",
    alpha = 1,
    foldid = inner_foldid,
    type.measure = "class",
    standardize = TRUE
  )
  lambda_lasso <- cv_lasso$lambda.min
  pred_lasso <- as.vector(predict(cv_lasso, newx = x_valid, s = "lambda.min", type = "class"))
  acc_lasso <- get_accuracy(y_valid, factor(pred_lasso, levels = class_levels))
  results <- rbind(
    results,
    data.frame(fold = fold, model = "lasso", accuracy = acc_lasso, tuned_lambda = lambda_lasso)
  )

  set.seed(seed + 200 + fold)
  cv_lasso_plus_sq <- cv.glmnet(
    x = x_train_plus_sq,
    y = y_train,
    family = "multinomial",
    alpha = 1,
    foldid = inner_foldid,
    type.measure = "class",
    standardize = TRUE
  )
  lambda_lasso_plus_sq <- cv_lasso_plus_sq$lambda.min
  pred_lasso_plus_sq <- as.vector(
    predict(cv_lasso_plus_sq, newx = x_valid_plus_sq, s = "lambda.min", type = "class")
  )
  acc_lasso_plus_sq <- get_accuracy(y_valid, factor(pred_lasso_plus_sq, levels = class_levels))
  results <- rbind(
    results,
    data.frame(
      fold = fold,
      model = "lasso_all_plus_squared",
      accuracy = acc_lasso_plus_sq,
      tuned_lambda = lambda_lasso_plus_sq
    )
  )

  set.seed(seed + 100 + fold)
  cv_ridge <- cv.glmnet(
    x = x_train,
    y = y_train,
    family = "multinomial",
    alpha = 0,
    foldid = inner_foldid,
    type.measure = "class",
    standardize = TRUE
  )
  lambda_ridge <- cv_ridge$lambda.min
  pred_ridge <- as.vector(predict(cv_ridge, newx = x_valid, s = "lambda.min", type = "class"))
  acc_ridge <- get_accuracy(y_valid, factor(pred_ridge, levels = class_levels))
  results <- rbind(
    results,
    data.frame(fold = fold, model = "ridge", accuracy = acc_ridge, tuned_lambda = lambda_ridge)
  )

  set.seed(seed + 300 + fold)
  cv_ridge_plus_sq <- cv.glmnet(
    x = x_train_plus_sq,
    y = y_train,
    family = "multinomial",
    alpha = 0,
    foldid = inner_foldid,
    type.measure = "class",
    standardize = TRUE
  )
  lambda_ridge_plus_sq <- cv_ridge_plus_sq$lambda.min
  pred_ridge_plus_sq <- as.vector(
    predict(cv_ridge_plus_sq, newx = x_valid_plus_sq, s = "lambda.min", type = "class")
  )
  acc_ridge_plus_sq <- get_accuracy(y_valid, factor(pred_ridge_plus_sq, levels = class_levels))
  results <- rbind(
    results,
    data.frame(
      fold = fold,
      model = "ridge_all_plus_squared",
      accuracy = acc_ridge_plus_sq,
      tuned_lambda = lambda_ridge_plus_sq
    )
  )
}

summary_df <- aggregate(accuracy ~ model, data = results, FUN = mean)
names(summary_df)[names(summary_df) == "accuracy"] <- "avg_accuracy"

lambda_df <- aggregate(tuned_lambda ~ model, data = results, FUN = function(v) {
  mean(v, na.rm = TRUE)
})
names(lambda_df)[names(lambda_df) == "tuned_lambda"] <- "avg_tuned_lambda"

summary_df <- merge(summary_df, lambda_df, by = "model", all.x = TRUE, sort = FALSE)

results_file <- file.path(output_dir, "cv_fold_results.csv")
summary_file <- file.path(output_dir, "cv_model_summary.csv")

write.csv(results, results_file, row.names = FALSE)
write.csv(summary_df, summary_file, row.names = FALSE)

message("Wrote fold metrics to: ", results_file)
message("Wrote model summary to: ", summary_file)
