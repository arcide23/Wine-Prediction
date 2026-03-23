#!/usr/bin/env Rscript

# Script to split processed wine dataset into train and test sets (80/20).

suppressWarnings(suppressMessages({
  # No external packages required; using base R only
}))

project_root <- getwd()

input_file <- file.path(project_root, "data", "raw", "output.csv")

if (!file.exists(input_file)) {
  stop("Input file not found at ", input_file, ". Run combine_wine_data.R first.")
}

message("Reading data from: ", input_file)

data <- read.csv(input_file, header = TRUE, stringsAsFactors = FALSE)

set.seed(123)  # for reproducibility

n <- nrow(data)
train_size <- floor(0.8 * n)

train_indices <- sample(seq_len(n), size = train_size)

train_data <- data[train_indices, , drop = FALSE]
test_data <- data[-train_indices, , drop = FALSE]

output_dir <- file.path(project_root, "data", "processed")
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
}

train_file <- file.path(output_dir, "train.csv")
test_file <- file.path(output_dir, "test.csv")

write.csv(train_data, train_file, row.names = FALSE)
write.csv(test_data, test_file, row.names = FALSE)

message("Train set written to: ", train_file)
message("Test set written to: ", test_file)
