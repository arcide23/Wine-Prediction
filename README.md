# Wine Quality Prediction Pipeline

This repository implements a reproducible modeling pipeline for predicting wine quality and documenting how model-selection decisions were made over time.

## Quick Start

```bash
make evaluate
```

`make evaluate` runs final test-set evaluation using the current training artifacts in `output/` and evaluates:
- the model with the best cross-validation average accuracy, and
- `random_forest_all_plus_squared` as an additional near-best benchmark.

If you need to regenerate training outputs from scratch before evaluating, run:

```bash
make split
make train
make graphs
make evaluate
```

You can also run:

```bash
make all
```

to execute `split`, `train`, and `graphs` in sequence (then run `make evaluate`).

## Project Goal

The goal is to predict wine quality from physicochemical features and compare multiple modeling approaches in a clean, replicable workflow.

## Process Narrative: From Regression to Classification

This project started with a regression framing (using RMSE thinking) because wine quality is ordinal and numeric. As we iterated, we shifted to a classification workflow for the final implementation:
- quality labels are treated as classes,
- models are trained for multiclass prediction, and
- model comparison is based on classification accuracy.

This transition was intentional: it made model behavior, fold-level comparisons, and final reporting easier to interpret consistently across model families in the current codebase.

## Data Pipeline

### 1) Data split
- Script: `scripts/split_wine_data.R`
- Reads raw data from `data/raw/output.csv`
- Creates an 80/20 train/test split
- Writes:
  - `data/processed/train.csv`
  - `data/processed/test.csv`
- Uses `set.seed(123)` for deterministic splitting.

### 2) Model training with 5-fold CV
- Script: `scripts/train_models.R`
- Trains and compares models on the training set using 5-fold cross-validation.
- Stores fold-level and summary-level outputs in `output/`.

### 3) Visualization of CV behavior
- Script: `scripts/create_graphs.R`
- Produces fold-accuracy boxplots (with outliers, outlier-removed, and top-3 subset).

### 4) Final test-set evaluation
- Script: `scripts/evaluate_best_model.R`
- Selects the best model by CV average accuracy from `output/cv_model_summary.csv`
- Evaluates that best model on `data/processed/test.csv`
- Also evaluates `random_forest_all_plus_squared` for direct side-by-side comparison.

## Models Tried

The current training pipeline compares these models:
- `intercept` (majority-class baseline)
- `all_parameters` (multinomial logistic regression, all raw features)
- `all_plus_squared` (adds squared feature terms)
- `all_plus_squared_interactions` (adds pairwise interactions and squared terms)
- `all_plus_squared_plus_interactions_plus_squared_interactions` (expanded interaction/squared-interaction specification)
- `lasso`
- `ridge`
- `lasso_all_plus_squared`
- `ridge_all_plus_squared`
- `random_forest`
- `random_forest_all_plus_squared`

## Evaluation Metric and Decision Process

The final codebase evaluates models primarily with **classification accuracy**.

Decision workflow:
1. `scripts/train_models.R` writes fold-level results to `output/cv_fold_results.csv`.
2. It aggregates fold results into `output/cv_model_summary.csv` (`avg_accuracy`, and average tuned lambda where applicable).
3. `scripts/create_graphs.R` visualizes fold distributions and outlier-robust comparisons.
4. `scripts/evaluate_best_model.R` picks the top CV model for test evaluation.
5. `make evaluate` additionally evaluates `random_forest_all_plus_squared` because it was extremely close to the best CV model and slightly better on held-out test performance in our run.

In the current output artifacts:
- Best CV average accuracy: `random_forest` (about `0.6650`)
- Close second by CV: `random_forest_all_plus_squared` (about `0.6640`)
- Test comparison from `make evaluate` run:
  - `random_forest`: about `0.7108`
  - `random_forest_all_plus_squared`: about `0.7138`

## Reproducibility and Sanity Checks

We included multiple reproducibility and robustness practices:
- Fixed random seeds in split, CV, and evaluation scripts.
- File existence checks before reading expected inputs.
- Output-directory creation checks before writing artifacts.
- Package checks for required libraries (including `randomForest`), with install guidance in error messages.
- Deterministic Makefile targets to run the same pipeline in the same order.

## Repository Organization

- `Makefile`: pipeline entry points (`split`, `train`, `graphs`, `evaluate`, `clean`)
- `scripts/`: data split, model training, graph generation, and final evaluation scripts
- `data/raw/`: source dataset
- `data/processed/`: train/test split outputs
- `output/`: cross-validation tables and PDF visualizations

## Team Workflow Note

During this assignment, most pushes were made under Daniel's account, even though work was collaborative across the team. For future projects, we plan to distribute commits more evenly so contribution history more clearly reflects shared development.
