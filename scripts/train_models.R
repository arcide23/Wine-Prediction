#!/usr/bin/env Rscript

suppressWarnings(suppressMessages({
  library(glmnet)
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
  mse = numeric(0),
  tuned_lambda = numeric(0),
  stringsAsFactors = FALSE
)

for (fold in seq_len(outer_folds)) {
  valid_idx <- which(fold_ids == fold)
  train_idx <- which(fold_ids != fold)

  x_train <- x_all[train_idx, , drop = FALSE]
  y_train <- y_all[train_idx]
  x_valid <- x_all[valid_idx, , drop = FALSE]
  y_valid <- y_all[valid_idx]

  # Intercept-only baseline uses mean response from fold-training split.
  pred_intercept <- rep(mean(y_train), length(valid_idx))
  mse_intercept <- mean((y_valid - pred_intercept)^2)
  results <- rbind(
    results,
    data.frame(fold = fold, model = "intercept", mse = mse_intercept, tuned_lambda = NA_real_)
  )

  lm_fit <- lm(y_train ~ ., data = as.data.frame(x_train))
  pred_lm <- predict(lm_fit, newdata = as.data.frame(x_valid))
  mse_lm <- mean((y_valid - pred_lm)^2)
  results <- rbind(
    results,
    data.frame(fold = fold, model = "all_parameters", mse = mse_lm, tuned_lambda = NA_real_)
  )

  train_fold_df <- train_df[train_idx, c(feature_cols, "quality"), drop = FALSE]
  valid_fold_df <- train_df[valid_idx, c(feature_cols, "quality"), drop = FALSE]

  squared_fit <- lm(squared_formula, data = train_fold_df)
  pred_squared <- predict(squared_fit, newdata = valid_fold_df)
  mse_squared <- mean((y_valid - pred_squared)^2)
  results <- rbind(
    results,
    data.frame(
      fold = fold,
      model = "all_plus_squared",
      mse = mse_squared,
      tuned_lambda = NA_real_
    )
  )

  squared_interactions_fit <- lm(squared_interactions_formula, data = train_fold_df)
  pred_squared_interactions <- predict(squared_interactions_fit, newdata = valid_fold_df)
  mse_squared_interactions <- mean((y_valid - pred_squared_interactions)^2)
  results <- rbind(
    results,
    data.frame(
      fold = fold,
      model = "all_plus_squared_interactions",
      mse = mse_squared_interactions,
      tuned_lambda = NA_real_
    )
  )

  full_squared_interactions_fit <- lm(full_squared_interactions_formula, data = train_fold_df)
  pred_full_squared_interactions <- predict(full_squared_interactions_fit, newdata = valid_fold_df)
  mse_full_squared_interactions <- mean((y_valid - pred_full_squared_interactions)^2)
  results <- rbind(
    results,
    data.frame(
      fold = fold,
      model = "all_plus_squared_plus_interactions_plus_squared_interactions",
      mse = mse_full_squared_interactions,
      tuned_lambda = NA_real_
    )
  )

  set.seed(seed + fold)
  cv_lasso <- cv.glmnet(
    x = x_train,
    y = y_train,
    alpha = 1,
    nfolds = inner_folds,
    standardize = TRUE
  )
  lambda_lasso <- cv_lasso$lambda.min
  pred_lasso <- as.numeric(predict(cv_lasso, newx = x_valid, s = "lambda.min"))
  mse_lasso <- mean((y_valid - pred_lasso)^2)
  results <- rbind(
    results,
    data.frame(fold = fold, model = "lasso", mse = mse_lasso, tuned_lambda = lambda_lasso)
  )

  set.seed(seed + 100 + fold)
  cv_ridge <- cv.glmnet(
    x = x_train,
    y = y_train,
    alpha = 0,
    nfolds = inner_folds,
    standardize = TRUE
  )
  lambda_ridge <- cv_ridge$lambda.min
  pred_ridge <- as.numeric(predict(cv_ridge, newx = x_valid, s = "lambda.min"))
  mse_ridge <- mean((y_valid - pred_ridge)^2)
  results <- rbind(
    results,
    data.frame(fold = fold, model = "ridge", mse = mse_ridge, tuned_lambda = lambda_ridge)
  )
}

summary_df <- aggregate(mse ~ model, data = results, FUN = mean)
names(summary_df)[names(summary_df) == "mse"] <- "avg_mse"

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
