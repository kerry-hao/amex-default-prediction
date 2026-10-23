# American Express Default Prediction

Predict future credit card default from a customer's monthly statements. This project turns statement histories into customer features, trains LightGBM with stratified cross-validation, and produces out-of-fold predictions, diagnostic charts, and a Kaggle submission.

The project is organized for learning the full workflow. Start with the notebooks and feature guide, then run each stage when your data are ready.

## Start here

1. Read [the workflow notebook](notebooks/01_workflow.ipynb) to follow the pipeline from statements to predictions.
2. Read [the feature and validation guide](docs/methodology.md) for the reasoning behind each step.
3. Browse [the results notebook](notebooks/02_results.ipynb) to see how the saved outputs will be read and interpreted.

## Dataset

Source: [American Express Default Prediction](https://www.kaggle.com/competitions/amex-default-prediction).

Each raw row describes one customer at one statement date. The label belongs to the customer. The competition defines the target over an 18-month performance window following the latest statement, using its 120-day payment default criterion. Each customer has up to 13 statements.

| Item | Reference data |
| --- | ---: |
| Training customers | 458,913 |
| Training statement rows | 5,531,451 |
| Test customers | 924,621 |
| Test statement rows | 11,363,762 |
| Statement predictors | 188 |
| Categorical predictors | 11 |
| Default labels in training | 118,828 |
| Default share in the sampled training data | 25.89% |

The counts come from the supplied EDA and LightGBM notebooks. `customer_ID` identifies customers, `S_2` is the statement date, and `target=1` denotes default. Predictor names describe five broad families:

| Prefix | Family |
| --- | --- |
| `D_*` | Delinquency |
| `S_*` | Spending, with `S_2` used as the statement date |
| `P_*` | Payment |
| `B_*` | Balance |
| `R_*` | Risk |

The features are anonymized and normalized. Interpret individual columns through their feature family and observed patterns. The competition sample retains 5% of the non-default class; the AMEX metric assigns those observations a weight of 20. Probability diagnostics in this project describe the sampled competition population.

## Workflow

| Stage | What the code does | Main output |
| --- | --- | --- |
| Prepare | Partition statements by customer and aggregate each history | Customer feature parquet files |
| Train | Fit five customer folds with early stopping | Models, OOF predictions, fold metrics |
| Report | Evaluate ranking and probability quality | Charts and `results.md` |
| Predict | Average fold margins and apply sigmoid | `submission.csv` |
| Profile | Summarize history lengths and feature availability | EDA tables and charts |

The baseline uses numerical **mean, minimum, maximum, and chronological last value**; categorical **last value and unique count**; and **statement count and history length**. This yields 732 features on the full official schema. An extended configuration adds standard deviations, first values, changes, and missing fractions.

All statements for a customer contribute to that customer's feature row. The five folds split those customer rows with stratification. Categories are fitted on the training rows of each fold and saved with that fold's model. Test inference reuses those category definitions.

## Reference result

| Model and source | Features | Validation result |
| --- | ---: | ---: |
| AmbrosM's supplied LightGBM Quickstart notebook | 469 | Mean five-fold AMEX 0.79383 |

That notebook uses Raddar's denoised integer Parquet data, a selected feature list, and 1,200 boosting rounds. Its five printed fold scores are recorded in [reference results](docs/reference_results.csv). This project's baseline uses the official CSV schema, all numerical predictors, shuffled stratified folds, and early stopping. Each project run records its own pooled OOF score and mean fold score in `metrics.json`.

The AMEX score combines normalized weighted Gini and default capture in the highest-ranked 4% of **cumulative observation weight**. Higher scores indicate better risk ranking. Additional outputs include ROC-AUC, average precision, log loss, Brier score, and calibration curves.

## Installation

Use Python 3.10 or later. Python 3.11 is used in CI.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebook]"
```

On Windows, activate with `.venv\Scripts\activate`. On macOS, install OpenMP with `brew install libomp` if your LightGBM installation needs it.

Preview the stages and resolved paths:

```bash
amex --config configs/baseline.json plan
```

This command reads the configuration and prints the workflow.

## Data setup and execution

When you are ready to work with the data, obtain the four official competition files from Kaggle and place them under `data/raw/`:

| File | Purpose |
| --- | --- |
| `train_data.csv` | Historical statements for training customers |
| `train_labels.csv` | Customer labels |
| `test_data.csv` | Historical statements for test customers |
| `sample_submission.csv` | Required customer IDs and submission order |

Run these commands from the project root:

```bash
amex --config configs/baseline.json prepare --split train
amex --config configs/baseline.json train
amex --config configs/baseline.json report
amex --config configs/baseline.json prepare --split test
amex --config configs/baseline.json predict
amex --config configs/baseline.json profile
```

Training creates its OOF evaluation as part of the same command. `predict` creates the submission file for later upload. Each command runs the stage named in the command.

For the trend features, use `configs/extended.json` in the same commands. It has separate feature and output directories. To rebuild an existing feature set or model run, add `--overwrite` to `prepare` or `train`. Choose a new experiment directory to retain earlier runs.

### Storage and memory

The official CSV collection is approximately 50 GB. Allow additional disk space for partitioned statements, prepared features, and models. A machine with 16-32 GB RAM is a practical starting point; actual requirements depend on feature recipe and training settings. Preparation uses 32 customer buckets and a 4 GB DuckDB memory setting. Training loads the aggregated customer table and fold matrices into RAM. Test prediction reads feature batches.

The pipeline accepts `.csv`, `.csv.gz`, and single `.parquet` statement files. To use another data representation, edit `paths.train` and `paths.test`. Configure missing sentinels in `data` to match that representation. The default `null` settings preserve the official CSV missing-value format. Keep the same representation for training and test.

## Project files

| Path | Contents |
| --- | --- |
| `configs/` | Baseline and extended experiments |
| `src/amex_default/features.py` | Disk partitioning and customer aggregation |
| `src/amex_default/data.py` | Label alignment and category handling |
| `src/amex_default/metrics.py` | AMEX metric and supporting metrics |
| `src/amex_default/training.py` | Customer folds and model artifacts |
| `src/amex_default/prediction.py` | Batched ensemble and submission alignment |
| `src/amex_default/reporting.py` | OOF charts and customer data profiles |
| `notebooks/` | English workflow and results walkthroughs |
| `tests/` | Metric, feature, alignment, and generated-fixture integration checks |

Run code checks with:

```bash
python -m compileall -q src
pytest -q
```

The tests generate small fixtures in temporary directories. GitHub Actions runs the same checks on pushes and pull requests.

## Outputs and next experiments

Training writes `oof.parquet`, `fold_metrics.csv`, `metrics.json`, `feature_importance.csv`, fold models, category maps, and `run.json`. The run manifest records configuration, package versions, feature schema, and feature preparation ID. Reporting creates discrimination, calibration, score distribution, and importance charts. Prediction writes `submission.csv` in the order of `sample_submission.csv`.

Useful next experiments are baseline versus extended features, last-value-only versus history summaries, feature-family ablations, and train/test missingness comparisons. Compare each experiment with the same customer folds and record both ranking and probability metrics.

## References

- [Competition and official data](https://www.kaggle.com/competitions/amex-default-prediction/data)
- [AmbrosM AMEX EDA which makes sense](https://www.kaggle.com/code/ambrosm/amex-eda-which-makes-sense)
- [AmbrosM AMEX LightGBM Quickstart](https://www.kaggle.com/code/ambrosm/amex-lightgbm-quickstart)
- [Raddar integer Parquet dataset](https://www.kaggle.com/datasets/raddar/amex-data-integer-dtypes-parquet-format)
- [LightGBM documentation](https://lightgbm.readthedocs.io/en/stable/)
- [DuckDB documentation](https://duckdb.org/docs/)

The supplied notebooks informed the workflow and reference summary. The project implements these steps as reusable Python modules. The MIT license covers the project code; Kaggle data follow the competition terms.
