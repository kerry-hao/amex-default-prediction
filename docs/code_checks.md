# Code checks

The release passes 12 automated tests on small generated fixtures. The tests cover the weighted AMEX calculation against an independent implementation, the weighted 4% cutoff, CSV and Parquet inputs, chronological endpoint values, missing endpoints, duplicate dates, label alignment, saved category vocabularies, and submission ID order. The integration fixture exercises feature preparation, three-fold training, saved-model prediction, reporting, and data profiling.

Python sources and notebook cells pass syntax checks. Both notebooks pass nbformat validation and are saved with empty execution counts and outputs. `amex plan` prints the configured workflow successfully.

| Package | Check environment version |
| --- | --- |
| Python | 3.12 |
| NumPy | 2.3.5 |
| pandas | 2.2.3 |
| PyArrow | 26.0.0 |
| DuckDB | 2.0.0 |
| LightGBM | 4.7.0 |
| scikit-learn | 1.8.0 |
| Matplotlib | 3.10.8 |

GitHub Actions installs the project with its development dependencies and runs the tests using Python 3.11. Each competition run records its actual package versions in `run.json`.
