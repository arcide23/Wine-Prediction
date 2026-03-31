#!/usr/bin/env Rscript

suppressWarnings(suppressMessages({
  library(glmnet)
  library(nnet)
}))

project_root <- getwd()
train_file <- file.path(project_root, "data", "processed", "train.csv")
test_file <- file.path(project_root, "data", "processed", "test.csv")
summary_file <- file.path(project_root, "output", "cv_model_summary.csv")

if (!file.exists(summary_file)) {
  stop("CV model summary not found at ", summary_file, ". Run make train first.")
}
if (!file.exists(train_file)) stop("Train file not found at ", train_file)
if (!file.exists(test_file)) stop("Test file not found at ", test_file)

seed <- 2026
set.seed(seed)

cv_summary <- read.csv(summary_file, header = TRUE, stringsAsFactors = FALSE)
if (!all(c("model", "avg_accuracy") %in% names(cv_summary))) {
  stop("Expected columns 'model' and 'avg_accuracy' in ", summary_file)
}

best_row <- cv_summary[which.max(cv_summary$avg_accuracy), , drop = FALSE]
best_model <- best_row$model[[1]]

train_df <- read.csv(train_file, header = TRUE, stringsAsFactors = FALSE)
test_df <- read.csv(test_file, header = TRUE, stringsAsFactors = FALSE)

if (!"quality" %in% names(train_df) || !"quality" %in% names(test_df)) {
  stop("Column 'quality' must exist in both train and test data.")
}

train_df$quality <- factor(train_df$quality)
class_levels <- levels(train_df$quality)
test_df$quality <- factor(test_df$quality, levels = class_levels)

feature_cols <- setdiff(names(train_df), "quality")

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

fit_predict_multinom <- function(formula_obj, train_data, test_data) {
  fit <- multinom(formula_obj, data = train_data, trace = FALSE, MaxNWts = 200000)
  pred <- predict(fit, newdata = test_data, type = "class")
  factor(pred, levels = class_levels)
}

fit_predict_glmnet_multinom <- function(x_train, y_train, x_test, alpha, fold_seed_offset = 0) {
  min_class_count <- min(table(y_train))
  inner_folds <- min(5, as.integer(min_class_count))
  while (inner_folds > 2 &&
    (min_class_count - ceiling(min_class_count / inner_folds)) < 2) {
    inner_folds <- inner_folds - 1
  }
  if (inner_folds < 2) {
    stop("Not enough observations per class to tune glmnet with CV on full training set.")
  }

  set.seed(seed + fold_seed_offset)
  foldid <- make_stratified_foldid(y_train, inner_folds)
  cvfit <- cv.glmnet(
    x = x_train,
    y = y_train,
    family = "multinomial",
    alpha = alpha,
    foldid = foldid,
    type.measure = "class",
    standardize = TRUE
  )

  pred <- as.vector(predict(cvfit, newx = x_test, s = "lambda.min", type = "class"))
  list(
    pred = factor(pred, levels = class_levels),
    lambda = cvfit$lambda.min
  )
}

# Build formulas used in training script
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
    paste(
      c(
        feature_cols,
        squared_terms,
        paste0("(", paste(feature_cols, collapse = " + "), ")^2"),
        squared_interaction_terms
      ),
      collapse = " + "
    )
  )
)

predict_for_model <- function(model_name) {
  pred <- NULL
  tuned_lambda <- NA_real_

  if (model_name == "intercept") {
    majority_class <- names(which.max(table(train_df$quality)))
    pred <- factor(rep(majority_class, nrow(test_df)), levels = class_levels)
  } else if (model_name == "all_parameters") {
    pred <- fit_predict_multinom(quality ~ ., train_df[, c(feature_cols, "quality"), drop = FALSE], test_df)
  } else if (model_name == "all_plus_squared") {
    pred <- fit_predict_multinom(squared_formula, train_df[, c(feature_cols, "quality"), drop = FALSE], test_df)
  } else if (model_name == "all_plus_squared_interactions") {
    pred <- fit_predict_multinom(
      squared_interactions_formula,
      train_df[, c(feature_cols, "quality"), drop = FALSE],
      test_df
    )
  } else if (model_name == "all_plus_squared_plus_interactions_plus_squared_interactions") {
    pred <- fit_predict_multinom(
      full_squared_interactions_formula,
      train_df[, c(feature_cols, "quality"), drop = FALSE],
      test_df
    )
  } else if (model_name %in% c("lasso", "ridge", "lasso_all_plus_squared", "ridge_all_plus_squared")) {
    x_train <- as.matrix(train_df[, feature_cols, drop = FALSE])
    x_test <- as.matrix(test_df[, feature_cols, drop = FALSE])

    if (model_name %in% c("lasso_all_plus_squared", "ridge_all_plus_squared")) {
      x_train_sq <- x_train^2
      x_test_sq <- x_test^2
      colnames(x_train_sq) <- paste0(colnames(x_train), "_sq")
      colnames(x_test_sq) <- paste0(colnames(x_test), "_sq")
      x_train <- cbind(x_train, x_train_sq)
      x_test <- cbind(x_test, x_test_sq)
    }

    alpha <- if (grepl("^lasso", model_name)) 1 else 0
    res <- fit_predict_glmnet_multinom(x_train, train_df$quality, x_test, alpha, fold_seed_offset = 900)
    pred <- res$pred
    tuned_lambda <- res$lambda
  } else if (model_name %in% c("random_forest", "random_forest_all_plus_squared")) {
    if (!requireNamespace("randomForest", quietly = TRUE)) {
      stop(
        "Model '", model_name, "' requires package 'randomForest' but it is not installed.\n",
        "Install it with:\n",
        "  install.packages('randomForest')\n"
      )
    }
    suppressWarnings(suppressMessages(library(randomForest)))

    train_rf <- train_df[, c(feature_cols, "quality"), drop = FALSE]
    test_rf <- test_df[, c(feature_cols, "quality"), drop = FALSE]

    if (model_name == "random_forest_all_plus_squared") {
      train_rf <- add_squared_columns(train_rf, feature_cols)
      test_rf <- add_squared_columns(test_rf, feature_cols)
    }

    set.seed(seed + 777)
    rf_fit <- randomForest(quality ~ ., data = train_rf)
    pred <- predict(rf_fit, newdata = test_rf, type = "class")
    pred <- factor(pred, levels = class_levels)
  } else {
    stop("Unknown model name: ", model_name)
  }

  list(
    pred = pred,
    tuned_lambda = tuned_lambda,
    test_accuracy = get_accuracy(test_df$quality, pred)
  )
}

best_res <- predict_for_model(best_model)

cat("Best model (by CV avg_accuracy):", best_model, "\n")
if (!is.na(best_res$tuned_lambda)) cat("Tuned lambda (train CV):", best_res$tuned_lambda, "\n")
cat("Final test accuracy:", best_res$test_accuracy, "\n")

extra_model <- "random_forest_all_plus_squared"
if (extra_model != best_model) {
  cat("\nAlso evaluating:", extra_model, "\n")
  extra_res <- tryCatch(
    predict_for_model(extra_model),
    error = function(e) list(error = conditionMessage(e))
  )
  if (!is.null(extra_res$error)) {
    cat("Skipped:", extra_res$error, "\n")
  } else {
    cat("Final test accuracy (", extra_model, "): ", extra_res$test_accuracy, "\n", sep = "")
  }
}

