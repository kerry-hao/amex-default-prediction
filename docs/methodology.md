# Features and validation

## From statements to one customer row

Consider three chronological values of a generic normalized numerical feature: 0.2, 0.4, and 0.8. The mean describes the typical level, the minimum and maximum summarize the range, and the last value describes the customer's latest observed state. The extended recipe also records the change from first to last and the distance of the last value from the historical mean. This is a teaching example.

| Feature suffix | Definition | Missing-value behavior |
| --- | --- | --- |
| `__mean` | Arithmetic mean over statements | Uses observed values |
| `__min`, `__max` | Lowest and highest value | Uses observed values |
| `__last` | Value on the latest statement date | Preserves a missing endpoint |
| `__std` | Sample standard deviation, extended recipe | Missing for fewer than two observed values |
| `__first` | Value on the earliest date, extended recipe | Preserves a missing endpoint |
| `__last_minus_first` | Last minus first, extended recipe | Missing when either endpoint is missing |
| `__last_minus_mean` | Last minus historical mean, extended recipe | Missing when last is missing |
| `__missing_fraction` | Missing statements divided by statements, extended recipe | All missing gives one |
| Categorical `__nunique` | Number of distinct observed categories | All missing gives zero |

The 11 categorical inputs are `B_30`, `B_38`, `D_114`, `D_116`, `D_117`, `D_120`, `D_126`, `D_63`, `D_64`, `D_66`, and `D_68`. Their latest value is passed to LightGBM as a categorical feature. Their unique counts are numerical features.

`n_statements` counts a customer's statements. `history_days` is the day difference between first and last dates. `S_2` orders the history and supplies that span. `customer_ID` joins customers to labels. The feature discovery rule selects named statement predictors and excludes an embedded `target` column automatically.

With the official schema, baseline features total `177 * 4 + 11 * 2 + 2 = 732`; extended features total `177 * 9 + 11 * 2 + 2 = 1617`.

## Processing large inputs

Preparation reads the source with DuckDB, normalizes data types, and writes temporary Parquet partitions using a hash of `customer_ID`. Every statement for the same customer goes into the same partition. Each partition is aggregated separately. Partitioned statement files are removed after feature preparation.

Date-based `arg_max_null` and `arg_min_null` select the actual endpoint values, including missing endpoints. Unique customer/date pairs make those endpoints unambiguous. The prepared feature files store numerical aggregates as float32.

## Customer folds

The training table contains one row per customer. Labels are joined by `customer_ID`, with exact ID coverage and binary label checks. `StratifiedKFold` creates five shuffled folds with seed 42. Each customer's OOF prediction comes from a model fitted on the other four folds.

Numerical aggregation uses only that customer's observed history. Category vocabularies are fitted separately on each fold's training rows. Missing and previously unseen categories are passed to LightGBM as missing category values. Model files and category maps are stored together.

The model uses AMEX for early stopping. Numerical missing values remain available to LightGBM's learned missing-value splits. The model is trained with the sampled competition labels and ordinary binary loss. The metric's 20x negative weight is applied inside AMEX scoring.

The supplied EDA records final training statements in March 2018 and later test cohorts ending in April and October 2019. The local folds assess discrimination across customers from the available labelled cohort. Train/test feature availability can be inspected separately with `profile`; `feature_missingness.csv` reports the difference in missing fractions. B_29 is a useful starting example from the EDA.

## Scores and probabilities

AMEX is the mean of normalized weighted Gini and default capture at a 4% weighted cutoff. Sort by decreasing score, give defaults weight 1 and non-defaults weight 20, then accumulate the weights. Capture is the fraction of defaults inside the first 4% of total weight. Normalized Gini evaluates the full ordering.

The implementation uses a stable descending sort for tied scores. Pooled OOF AMEX uses all OOF predictions together. Mean fold AMEX averages scores computed within the folds. Both are saved because the cutoff makes them slightly different summaries.

ROC-AUC and average precision describe discrimination. Log loss and Brier score evaluate predicted probabilities. Calibration plots compare mean predictions with observed frequencies in quantile bins. These probabilities and frequencies refer to the sampled competition population, whose default share is approximately 25.89%.

For test predictions, the pipeline averages the fold models' raw margins and applies sigmoid. This follows the ensemble idea in the supplied Quickstart while exporting values in the probability range. Sigmoid preserves the averaged margin's ordering. Submission rows are joined to `sample_submission.csv` by ID and returned in its original order.

## Reference evidence

| Source file | Evidence used |
| --- | --- |
| `amex-eda-which-makes-sense.ipynb` | Cell 5 label counts; cell 7 shapes; cells 9 and 19 time coverage; cell 17 history lengths; cell 25 B_29 availability |
| `amex-lightgbm-quickstart.ipynb` | Cell 4 feature table sizes; cell 6 training parameters and fold scores; cells 11-12 mean-margin ensemble |

Cell numbers above follow the supplied files' zero-based cell order. The Quickstart result is the mean of its five printed fold scores. [reference_results.csv](reference_results.csv) preserves those values. Each new run creates its own scores and plots under its experiment directory.
