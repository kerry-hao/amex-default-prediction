import numpy as np
import pandas as pd
import pytest

from amex_default.config import load_config
from amex_default.data import align_labels, category_maps, model_frame
from amex_default.features import prepare
from amex_default.prediction import align_submission


def config_for(tmp_path):
    from pathlib import Path
    cfg = load_config(Path(__file__).parents[1] / "configs" / "baseline.json", root=tmp_path)
    cfg["processing"].update(partitions=2, threads=1, memory_limit="512MB")
    (tmp_path / "data" / "raw").mkdir(parents=True, exist_ok=True)
    return cfg


@pytest.mark.parametrize("extension", ["csv", "parquet"])
def test_chronological_last_preserves_missing_and_target_is_excluded(tmp_path, extension):
    cfg = config_for(tmp_path)
    cfg["features"]["recipe"] = "extended"
    cfg["paths"]["train"] = "data/raw/statements." + extension
    frame = pd.DataFrame({"customer_ID": ["a", "b", "a", "b"],
                          "S_2": ["2020-02-01", "2020-02-01", "2020-01-01", "2020-01-01"],
                          "P_2": [np.nan, 0.4, 0.9, 0.2], "D_63": [None, "B", "A", "B"],
                          "target": [1, 0, 1, 0]})
    path = tmp_path / cfg["paths"]["train"]
    if extension == "csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)
    prepare(cfg, "train")
    result = pd.read_parquet(tmp_path / "data/features/baseline/train").set_index("customer_ID")
    assert pd.isna(result.loc["a", "P_2__last"])
    assert pd.isna(result.loc["a", "D_63__last"])
    assert result.loc["a", "P_2__mean"] == pytest.approx(0.9)
    assert result.loc["a", "P_2__missing_fraction"] == pytest.approx(0.5)
    assert result.loc["b", "P_2__last_minus_first"] == pytest.approx(0.2)
    assert result.loc["a", "n_statements"] == 2
    assert result.loc["a", "history_days"] == 31
    assert not any("target" in c for c in result.columns)


def test_duplicate_statement_dates_fail(tmp_path):
    cfg = config_for(tmp_path)
    pd.DataFrame({"customer_ID": ["a", "a"], "S_2": ["2020-01-01"]*2,
                  "P_2": [0.1, 0.2]}).to_csv(tmp_path / cfg["paths"]["train"], index=False)
    with pytest.raises(ValueError, match="unique"):
        prepare(cfg, "train")


def test_label_join_and_submission_order():
    features = pd.DataFrame({"customer_ID": ["b", "a"]})
    labels = pd.DataFrame({"customer_ID": ["a", "b"], "target": [1, 0]})
    assert align_labels(features, labels).tolist() == [0, 1]
    predictions = pd.DataFrame({"customer_ID": ["b", "a"], "prediction": [0.1, 0.8]})
    result = align_submission(predictions, pd.DataFrame({"customer_ID": ["a", "b"]}))
    assert result.prediction.tolist() == [0.8, 0.1]
    with pytest.raises(ValueError):
        align_labels(features, labels.iloc[:1])
    with pytest.raises(ValueError):
        align_submission(predictions, pd.DataFrame({"customer_ID": ["a", "a"]}))


def test_categories_fit_on_training_rows():
    train = pd.DataFrame({"D_63__last": ["A", None], "P_2__last": [0.2, np.nan]})
    valid = pd.DataFrame({"D_63__last": ["NEW"], "P_2__last": [0.4]})
    maps = category_maps(train, ["D_63__last"])
    encoded = model_frame(valid, list(train.columns), maps)
    assert maps == {"D_63__last": ["A"]}
    assert pd.isna(encoded.loc[0, "D_63__last"])
